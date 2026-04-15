"""Orchestrates single-row inference: raw input → feature engineering → predict → log."""

import logging

import pandas as pd
from sqlalchemy.orm import Session

from app.ml.pipeline import FEATURE_COLUMNS, engineer_features
from app.ml.registry import load_model
from app.models.prediction import InferenceInput, Prediction
from app.repositories import PredictionRepository
from app.schemas.inference import PredictRequest

logger = logging.getLogger(__name__)


def _request_to_raw_df(request: PredictRequest) -> pd.DataFrame:
    """Convert a PredictRequest into a single-row DataFrame with CSV-like column names."""
    data = {
        "Year_Birth": [request.year_birth],
        "Education": [request.education],
        "Marital_Status": [request.marital_status],
        "Income": [request.income],
        "Kidhome": [request.kidhome],
        "Teenhome": [request.teenhome],
        "Dt_Customer": [request.dt_customer],
        "Recency": [request.recency],
        "MntWines": [request.mnt_wines],
        "MntFruits": [request.mnt_fruits],
        "MntMeatProducts": [request.mnt_meat_products],
        "MntFishProducts": [request.mnt_fish_products],
        "MntSweetProducts": [request.mnt_sweet_products],
        "MntGoldProds": [request.mnt_gold_prods],
        "NumDealsPurchases": [request.num_deals_purchases],
        "NumWebPurchases": [request.num_web_purchases],
        "NumCatalogPurchases": [request.num_catalog_purchases],
        "NumStorePurchases": [request.num_store_purchases],
        "NumWebVisitsMonth": [request.num_web_visits_month],
        "Complain": [request.complain],
    }
    return pd.DataFrame(data)


class InferenceService:
    def __init__(self, db: Session):
        self.db = db
        self.prediction_repo = PredictionRepository(db)

    def predict(self, request: PredictRequest) -> dict:
        """Run a single-row prediction and persist the result to the database."""

        df_raw = _request_to_raw_df(request)
        logger.info("Raw DF values:\n%s", df_raw.iloc[0].to_dict())

        df_fe = engineer_features(df_raw)
        X = df_fe[FEATURE_COLUMNS]

        nan_cols = X.columns[X.isna().any()].tolist()
        if nan_cols:
            logger.error("NaN detected in columns: %s", nan_cols)

        pipeline, version = load_model(request.model_version)

        pred_class = int(pipeline.predict(X)[0])
        pred_proba = float(pipeline.predict_proba(X)[0, 1])

        with self.db.begin():
            prediction_row = Prediction(
                prediction=pred_class,
                prediction_proba=round(pred_proba, 6),
                source="inference",
                model_version=version,
            )
            self.prediction_repo.add(prediction_row)

            inference_input = InferenceInput(
                prediction=prediction_row,
                **request.model_dump(exclude={"model_version"}),
            )
            self.prediction_repo.add_inference_input(inference_input)

        logger.info(
            "Inference prediction=%d proba=%.4f version=%s",
            pred_class,
            pred_proba,
            version,
        )

        return {
            "prediction_id": prediction_row.id,
            "prediction": pred_class,
            "prediction_proba": round(pred_proba, 6),
            "model_version": version,
        }

    def update_true_label(self, prediction_id: int, true_label: int) -> dict | None:
        """Update the true label of a prediction."""

        with self.db.begin():
            updated = self.prediction_repo.update_true_label(prediction_id, true_label)

            if not updated:
                return None

        logger.info(
            "Updated prediction id=%d with true_label=%d",
            prediction_id,
            true_label,
        )

        return {
            "prediction_id": prediction_id,
            "true_label": true_label,
            "message": f"Prediction {prediction_id} updated with true_label={true_label}",
        }
