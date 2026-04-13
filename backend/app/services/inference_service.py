"""Orchestrates single-row inference: raw input → feature engineering → predict → log."""

import logging

import pandas as pd
from sqlalchemy.orm import Session

from app.ml.pipeline import FEATURE_COLUMNS, engineer_features
from app.ml.registry import load_model
from app.models.prediction import InferenceInput, Prediction
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


def predict(request: PredictRequest, db: Session) -> dict:
    """Run a single-row prediction and persist the result to the database.

    Steps
    -----
    1. Convert the raw request payload to a DataFrame.
    2. Apply the same feature engineering used during training.
    3. Load the latest trained pipeline from disk.
    4. Generate prediction class and probability.
    5. Persist a Prediction row and the raw InferenceInput payload.

    Returns
    -------
    dict
        Keys: ``prediction``, ``prediction_proba``, ``model_version``.
    """
    df_raw = _request_to_raw_df(request)
    logger.info("Raw DF dtypes:\n%s", df_raw.dtypes)
    logger.info("Raw DF values:\n%s", df_raw.iloc[0].to_dict())

    df_fe = engineer_features(df_raw)
    X = df_fe[FEATURE_COLUMNS]

    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        logger.error("NaN detected in columns: %s", nan_cols)
        logger.error("Feature values:\n%s", X.iloc[0].to_dict())

    pipeline, version = load_model()

    pred_class = int(pipeline.predict(X)[0])
    pred_proba = float(pipeline.predict_proba(X)[0, 1])

    prediction_row = Prediction(
        prediction=pred_class,
        prediction_proba=round(pred_proba, 6),
        source="inference",
        model_version=version,
    )
    db.add(prediction_row)
    db.flush()

    inference_input = InferenceInput(
        prediction_id=prediction_row.id,
        **request.model_dump(),
    )
    db.add(inference_input)
    db.commit()

    logger.info(
        "Inference prediction=%d proba=%.4f version=%s",
        pred_class,
        pred_proba,
        version,
    )

    return {
        "prediction": pred_class,
        "prediction_proba": round(pred_proba, 6),
        "model_version": version,
    }
