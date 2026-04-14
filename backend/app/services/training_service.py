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
from app.models.customer import Customer
from app.models.customer_feature import CustomerFeature
from app.models.lookup import EducationLevel, MaritalStatus
from app.models.prediction import Prediction
from app.models.training_result import TrainingResult

logger = logging.getLogger(__name__)


def _load_raw_customers(db: Session) -> pd.DataFrame:
    """Query the customers table and return a raw DataFrame with CSV-like column names."""
    rows = (
        db.query(
            Customer.id.label("Id"),
            Customer.year_birth.label("Year_Birth"),
            EducationLevel.name.label("Education"),
            MaritalStatus.name.label("Marital_Status"),
            Customer.income.label("Income"),
            Customer.kidhome.label("Kidhome"),
            Customer.teenhome.label("Teenhome"),
            Customer.dt_customer.label("Dt_Customer"),
            Customer.recency.label("Recency"),
            Customer.mnt_wines.label("MntWines"),
            Customer.mnt_fruits.label("MntFruits"),
            Customer.mnt_meat_products.label("MntMeatProducts"),
            Customer.mnt_fish_products.label("MntFishProducts"),
            Customer.mnt_sweet_products.label("MntSweetProducts"),
            Customer.mnt_gold_prods.label("MntGoldProds"),
            Customer.num_deals_purchases.label("NumDealsPurchases"),
            Customer.num_web_purchases.label("NumWebPurchases"),
            Customer.num_catalog_purchases.label("NumCatalogPurchases"),
            Customer.num_store_purchases.label("NumStorePurchases"),
            Customer.num_web_visits_month.label("NumWebVisitsMonth"),
            Customer.recency.label("Recency_dup"),
            Customer.complain.label("Complain"),
            Customer.response.label("Response"),
        )
        .join(EducationLevel, Customer.education_level_id == EducationLevel.id)
        .join(MaritalStatus, Customer.marital_status_id == MaritalStatus.id)
        .all()
    )

    df = pd.DataFrame(rows)
    if "Recency_dup" in df.columns:
        df.drop(columns=["Recency_dup"], inplace=True)

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"]).dt.strftime("%m/%d/%Y")
    df["Complain"] = df["Complain"].astype(int)
    df["Response"] = df["Response"].astype(int)

    return df


def train_model(db: Session) -> dict:
    """Run the full training pipeline and persist the results.

    Steps
    -----
    1. Load raw customer data from the database.
    2. Clean the data (deduplicate, remove outliers and invalid statuses).
    3. Stratified train/test split.
    4. Apply feature engineering independently to each split.
    5. Build and fit the sklearn pipeline via ModelFactory.
    6. Evaluate on the test set; compute all classification metrics.
    7. Save the fitted pipeline artefact to disk.
    8. Persist a TrainingResult row to the database.
    9. Predict on the training split and bulk-insert Prediction rows.

    Returns
    -------
    dict
        A dictionary containing split sizes, all evaluation metrics,
        the saved model version, and a human-readable message.
    """
    df_raw = _load_raw_customers(db)
    logger.info("Loaded %d raw customer rows from DB", len(df_raw))

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

    # Зберігаємо customer IDs до feature engineering (яке їх видалить)
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

    result_row = TrainingResult(
        model_version=version,
        train_size=len(X_train),
        test_size=len(X_test),
        accuracy=metrics["accuracy"],
        balanced_accuracy=metrics["balanced_accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"],
        roc_auc=metrics["roc_auc"],
        pr_auc=metrics["pr_auc"],
        mcc=metrics["mcc"],
        true_negatives=metrics["true_negatives"],
        false_positives=metrics["false_positives"],
        false_negatives=metrics["false_negatives"],
        true_positives=metrics["true_positives"],
    )
    db.add(result_row)
    db.commit()
    logger.info("Training result persisted for version: %s", version)

    train_preds = pipeline.predict(X_train)
    train_probas = pipeline.predict_proba(X_train)[:, 1]

    cf_id_map: dict[int, int] = {}
    cf_response_map: dict[int, int] = {}
    if train_customer_ids:
        cf_rows = (
            db.query(CustomerFeature.id, CustomerFeature.customer_id, CustomerFeature.response)
            .filter(CustomerFeature.customer_id.in_(train_customer_ids))
            .all()
        )
        cf_id_map = {row.customer_id: row.id for row in cf_rows}
        cf_response_map = {row.customer_id: row.response for row in cf_rows}
        logger.info(
            "Mapped %d / %d train customers to customer_features",
            len(cf_id_map),
            len(train_customer_ids),
        )

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
            train_preds, train_probas, train_customer_ids or [None] * len(train_preds)
        )
    ]
    db.bulk_save_objects(prediction_rows)
    db.commit()
    logger.info("Saved %d train predictions to DB", len(prediction_rows))

    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        **metrics,
        "model_version": version,
        "message": (
            f"Model trained and saved. {len(prediction_rows)} train predictions stored."
        ),
    }


def list_training_results(db: Session, limit: int = 20) -> list[TrainingResult]:
    """Return the most recent training results ordered by creation time descending.

    Parameters
    ----------
    db:
        Active SQLAlchemy session.
    limit:
        Maximum number of rows to return.
    """
    return (
        db.query(TrainingResult)
        .order_by(TrainingResult.created_at.desc())
        .limit(limit)
        .all()
    )


def get_feature_importance() -> dict:
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
        model, version = load_model()
    except FileNotFoundError:
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
