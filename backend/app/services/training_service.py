"""Orchestrates model training: load → clean → split → engineer → fit → evaluate → save → persist."""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from app.core.config import settings
from app.ml.pipeline import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ModelFactory,
    clean_raw_data,
    engineer_features,
)
from app.ml.registry import load_model, save_model
from app.models.prediction import Prediction
from app.models.training_result import TrainingResult
from app.repositories import (
    CustomerFeatureRepository,
    CustomerRepository,
    PredictionRepository,
    TrainingResultRepository,
)

logger = logging.getLogger(__name__)


class TrainingService:
    def __init__(self, db: Session):
        self.db = db
        self.customer_repo = CustomerRepository(db)
        self.customer_feature_repo = CustomerFeatureRepository(db)
        self.prediction_repo = PredictionRepository(db)
        self.training_result_repo = TrainingResultRepository(db)

    def _load_raw_customers(self) -> pd.DataFrame:
        """Query the customers table and return a raw DataFrame with CSV-like column names."""
        rows = self.customer_repo.get_raw_customers_with_lookups()

        df = pd.DataFrame(rows)
        if "Recency_dup" in df.columns:
            df.drop(columns=["Recency_dup"], inplace=True)

        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"]).dt.strftime("%m/%d/%Y")
        df["Complain"] = df["Complain"].astype(int)
        df["Response"] = df["Response"].astype(int)

        return df

    def train_model(self) -> dict:
        """Run full training pipeline in a single DB transaction."""

        df_raw = self._load_raw_customers()
        logger.info("Loaded %d raw customer rows", len(df_raw))

        df_clean = clean_raw_data(df_raw)
        logger.info("After cleaning: %d rows", len(df_clean))

        df_train, df_test = train_test_split(
            df_clean,
            test_size=settings.TEST_SIZE,
            random_state=settings.RANDOM_STATE,
            stratify=df_clean["Response"],
        )

        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        train_customer_ids = df_train["Id"].tolist() if "Id" in df_train.columns else []

        fe_train = engineer_features(df_train)
        fe_test = engineer_features(df_test)

        X_train = fe_train[FEATURE_COLUMNS]
        y_train = fe_train[TARGET_COLUMN]
        X_test = fe_test[FEATURE_COLUMNS]
        y_test = fe_test[TARGET_COLUMN]

        pipeline = ModelFactory.build(settings)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "pr_auc": round(average_precision_score(y_test, y_proba), 4),
            "mcc": round(matthews_corrcoef(y_test, y_pred), 4),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

        logger.info("Test metrics: %s", metrics)

        version = save_model(pipeline)
        logger.info("Model saved: %s", version)

        train_preds = pipeline.predict(X_train)
        train_probas = pipeline.predict_proba(X_train)[:, 1]

        cf_id_map = {}
        cf_response_map = {}

        if train_customer_ids:
            cf_rows = self.customer_feature_repo.get_by_customer_ids(train_customer_ids)
            cf_id_map = {r.customer_id: r.id for r in cf_rows}
            cf_response_map = {r.customer_id: r.response for r in cf_rows}

        prediction_rows = [
            Prediction(
                prediction=int(pred),
                prediction_proba=float(proba),
                true_label=cf_response_map.get(cust_id),
                source="train",
                model_version=version,
                customer_feature_id=cf_id_map.get(cust_id),
            )
            for pred, proba, cust_id in zip(
                train_preds,
                train_probas,
                train_customer_ids or [None] * len(train_preds),
            )
        ]
        
        with self.db.begin():

            result_row = TrainingResult(
                model_version=version,
                train_size=len(X_train),
                test_size=len(X_test),
                **metrics,
            )
            self.training_result_repo.add(result_row)

            self.prediction_repo.bulk_save(prediction_rows)

        logger.info("Training pipeline committed for version: %s", version)

        return {
            "train_size": len(X_train),
            "test_size": len(X_test),
            **metrics,
            "model_version": version,
            "message": f"Model trained. {len(prediction_rows)} predictions stored.",
        }

    def list_training_results(self, limit: int = 20) -> list[TrainingResult]:
        """Return the most recent training results ordered by creation time descending.

        Parameters
        ----------
        limit:
            Maximum number of rows to return.
        """
        return self.training_result_repo.get_recent(limit)

    def get_feature_importance(self, model_version: str | None = None) -> dict:
        """Calculate feature importance using all available methods.

        Returns coefficient-based importance for linear models and
        feature_importances_ for tree-based models.

        Returns
        -------
        dict
            Feature importance data with model_version, model_type,
            available_methods, unavailable_methods, and importance values.
        """
        try:
            model, version = load_model(version=model_version)
        except FileNotFoundError:
            if model_version:
                raise ValueError(f"No trained model found for version {model_version}.")
            raise ValueError("No trained model found. Run /train-model first.")

        model_type = type(model.named_steps["classifier"]).__name__
        feature_names = FEATURE_COLUMNS

        result = {
            "model_version": version,
            "model_type": model_type,
            "available_methods": [],
            "unavailable_methods": [],
            "coefficient_importance": None,
            "feature_importance": None,
        }

        classifier = model.named_steps["classifier"]

        # Coefficient-based (LogisticRegression, LinearSVM, etc.)
        if hasattr(classifier, "coef_"):
            coef = np.abs(classifier.coef_[0])
            indices = np.argsort(coef)[::-1]
            result["coefficient_importance"] = [
                {"feature": feature_names[idx], "importance": float(coef[idx]), "rank": rank}
                for rank, idx in enumerate(indices, 1)
            ]
            result["available_methods"].append("coefficient")
        else:
            result["unavailable_methods"].append({
                "method": "coefficient",
                "reason": "Only available for linear models (LogisticRegression, LinearSVM, etc.)",
            })

        # Feature importance (RandomForest, XGBoost, etc.)
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
            indices = np.argsort(importances)[::-1]
            result["feature_importance"] = [
                {"feature": feature_names[idx], "importance": float(importances[idx]), "rank": rank}
                for rank, idx in enumerate(indices, 1)
            ]
            result["available_methods"].append("feature_importance")
        else:
            result["unavailable_methods"].append({
                "method": "feature_importance",
                "reason": "Only available for tree-based models (RandomForest, XGBoost, etc.)",
            })

        return result
