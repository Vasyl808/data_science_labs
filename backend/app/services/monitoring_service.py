"""Business logic for ML model monitoring with Evidently AI 0.7.21.

Generates HTML reports for data drift, target drift, classification
performance, and data quality by comparing reference (training) data
with current (inference) data.

Reference data comes from predictions with source='train' where true_label
is set from CustomerFeature.response.

Current data comes from predictions with source='inference' where true_label
may be set after calling the client.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.ml.pipeline import (
    ALONE_STATUSES,
    EDUCATION_ORDER,
    FEATURE_COLUMNS,
    LOG_COLS,
    MNT_COLS,
    SNAPSHOT_DATE_STR,
)
from app.models.customer_feature import CustomerFeature
from app.models.prediction import InferenceInput, Prediction

logger = logging.getLogger(__name__)


_CF_TO_FEATURE: dict[str, str] = {
    "education": "Education",
    "income": "Income",
    "kidhome": "Kidhome",
    "teenhome": "Teenhome",
    "recency": "Recency",
    "num_store_purchases": "NumStorePurchases",
    "age": "Age",
    "customer_tenure_days": "Customer_Tenure_Days",
    "mnt_wines_log": "MntWines_log",
    "mnt_meat_products_log": "MntMeatProducts_log",
    "mnt_fruits_log": "MntFruits_log",
    "mnt_fish_products_log": "MntFishProducts_log",
    "mnt_sweet_products_log": "MntSweetProducts_log",
    "mnt_gold_prods_log": "MntGoldProds_log",
    "num_catalog_purchases_log": "NumCatalogPurchases_log",
    "num_web_purchases_log": "NumWebPurchases_log",
    "total_mnt": "TotalMnt",
    "total_purchases": "TotalPurchases",
    "wine_ratio": "WineRatio",
    "meat_ratio": "MeatRatio",
    "premium_ratio": "PremiumRatio",
    "catalog_share": "CatalogShare",
    "is_alone": "is_alone",
}

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"


def get_reference_data(session: Session) -> pd.DataFrame:
    """Load reference (training) data with true_label from predictions.

    Returns features + target (true_label) + timestamp for predictions
    where source='train'. True_label is taken from Prediction.true_label
    which was set from CustomerFeature.response during training.
    """
    rows = (
        session.query(
            CustomerFeature.education,
            CustomerFeature.income,
            CustomerFeature.kidhome,
            CustomerFeature.teenhome,
            CustomerFeature.recency,
            CustomerFeature.num_store_purchases,
            CustomerFeature.age,
            CustomerFeature.customer_tenure_days,
            CustomerFeature.mnt_wines_log,
            CustomerFeature.mnt_meat_products_log,
            CustomerFeature.mnt_fruits_log,
            CustomerFeature.mnt_fish_products_log,
            CustomerFeature.mnt_sweet_products_log,
            CustomerFeature.mnt_gold_prods_log,
            CustomerFeature.num_catalog_purchases_log,
            CustomerFeature.num_web_purchases_log,
            CustomerFeature.total_mnt,
            CustomerFeature.total_purchases,
            CustomerFeature.wine_ratio,
            CustomerFeature.meat_ratio,
            CustomerFeature.premium_ratio,
            CustomerFeature.catalog_share,
            CustomerFeature.is_alone,
            Prediction.prediction,
            Prediction.true_label,
            CustomerFeature.timestamp,
        )
        .join(
            Prediction,
            CustomerFeature.id == Prediction.customer_feature_id,
        )
        .filter(Prediction.source == "train")
        .filter(Prediction.true_label.isnot(None))
        .distinct()
        .all()
    )

    if not rows:
        logger.warning("No reference data found with true_label")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.rename(columns=_CF_TO_FEATURE, inplace=True)
    df.rename(columns={"true_label": "target", "prediction": "prediction"}, inplace=True)
    logger.info("Reference data loaded: %d rows", len(df))
    return df


def _engineer_inference_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to raw inference inputs.

    Mirrors app.ml.pipeline.engineer_features for column name and
    value distribution compatibility with reference data.
    """
    df = df_raw.copy()

    df["Dt_Customer"] = pd.to_datetime(df["dt_customer"], errors="coerce")
    snapshot = pd.Timestamp(SNAPSHOT_DATE_STR)
    df["Age"] = 2014 - df["year_birth"]
    df["Customer_Tenure_Days"] = (snapshot - df["Dt_Customer"]).dt.days

    rename_map = {
        "mnt_wines": "MntWines",
        "mnt_fruits": "MntFruits",
        "mnt_meat_products": "MntMeatProducts",
        "mnt_fish_products": "MntFishProducts",
        "mnt_sweet_products": "MntSweetProducts",
        "mnt_gold_prods": "MntGoldProds",
        "num_web_purchases": "NumWebPurchases",
        "num_catalog_purchases": "NumCatalogPurchases",
        "num_store_purchases": "NumStorePurchases",
        "income": "Income",
        "kidhome": "Kidhome",
        "teenhome": "Teenhome",
        "recency": "Recency",
        "education": "Education",
        "marital_status": "Marital_Status",
    }
    df.rename(columns=rename_map, inplace=True)

    df["TotalMnt"] = df[MNT_COLS].sum(axis=1)
    df["TotalPurchases"] = df[
        ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    ].sum(axis=1)

    df["WineRatio"] = df["MntWines"] / (df["Income"] + 1)
    df["MeatRatio"] = df["MntMeatProducts"] / (df["Income"] + 1)
    df["PremiumRatio"] = (df["MntWines"] + df["MntMeatProducts"]) / (
        df["TotalMnt"] + 1
    )
    df["CatalogShare"] = df["NumCatalogPurchases"] / (df["TotalPurchases"] + 1)

    for col in LOG_COLS:
        df[f"{col}_log"] = np.log1p(df[col])

    df["Education"] = df["Education"].map(EDUCATION_ORDER)

    df["is_alone"] = df["Marital_Status"].isin(ALONE_STATUSES).astype(int)

    return df


def get_current_data(session: Session) -> pd.DataFrame:
    """Load current (inference) data with prediction and true_label.

    Returns features + prediction + true_label (if set) + timestamp.
    true_label is set after calling the client via PATCH endpoint.
    """
    rows = (
        session.query(
            InferenceInput.year_birth,
            InferenceInput.education,
            InferenceInput.marital_status,
            InferenceInput.income,
            InferenceInput.kidhome,
            InferenceInput.teenhome,
            InferenceInput.dt_customer,
            InferenceInput.recency,
            InferenceInput.mnt_wines,
            InferenceInput.mnt_fruits,
            InferenceInput.mnt_meat_products,
            InferenceInput.mnt_fish_products,
            InferenceInput.mnt_sweet_products,
            InferenceInput.mnt_gold_prods,
            InferenceInput.num_web_purchases,
            InferenceInput.num_catalog_purchases,
            InferenceInput.num_store_purchases,
            InferenceInput.created_at,
            Prediction.prediction,
            Prediction.true_label,
        )
        .join(Prediction, InferenceInput.prediction_id == Prediction.id)
        .filter(Prediction.source == "inference")
        .all()
    )

    if not rows:
        logger.warning("No inference data found")
        return pd.DataFrame()

    df_raw = pd.DataFrame(rows)
    df_fe = _engineer_inference_features(df_raw)

    result = df_fe[FEATURE_COLUMNS].copy()
    result["prediction"] = df_raw["prediction"].values
    result["true_label"] = df_raw["true_label"].values
    result["timestamp"] = df_raw["created_at"].values

    logger.info("Current data loaded: %d rows (%d with true_label)",
                len(result), result["true_label"].notna().sum())
    return result


def _safe_feature_cols(
    feature_cols: list[str],
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> list[str]:
    """Return only columns present in BOTH DataFrames."""
    ref_set = set(reference.columns)
    cur_set = set(current.columns)
    return [c for c in feature_cols if c in ref_set and c in cur_set]


def _html(snapshot) -> str:
    """Render Evidently Snapshot to HTML string."""
    return snapshot.get_html_str(as_iframe=False)


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> str:
    """Data Drift Report - compares feature distributions."""
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    feature_cols = [c for c in FEATURE_COLUMNS if c in current.columns]

    if reference.empty:
        logger.warning("data_drift: no reference — using DataSummaryPreset")
        report = Report(metrics=[DataSummaryPreset()])
        snapshot = report.run(current_data=current[feature_cols])
    else:
        cols = [c for c in feature_cols if c in reference.columns]
        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(
            current_data=current[cols],
            reference_data=reference[cols],
        )

    return snapshot.get_html_str(as_iframe=False)


def generate_target_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> str:
    """Target Drift Report - compares target distribution.

    For current data, uses true_label if available, otherwise prediction.
    """
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    current_with_labels = current[current["true_label"].notna()]

    if current_with_labels.empty:
        logger.warning("target_drift: no true_label in current — using prediction")
        current_target = current[["prediction"]].rename(columns={"prediction": "target"})
    else:
        current_target = current_with_labels[["true_label"]].rename(columns={"true_label": "target"})

    if reference.empty:
        logger.warning("target_drift: no reference — using DataSummaryPreset")
        report = Report(metrics=[DataSummaryPreset()])
        snapshot = report.run(current_data=current_target)
    else:
        reference_target = reference[["target"]]
        report = Report(metrics=[DataDriftPreset(columns=["target"])])
        snapshot = report.run(
            current_data=current_target,
            reference_data=reference_target,
        )

    return snapshot.get_html_str(as_iframe=False)


def generate_classification_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> str:
    """Classification Performance Report.

    Requires true_label in current data (set after calling client).
    Uses DataDefinition with BinaryClassification for Evidently 0.7.21.
    """
    from evidently import Dataset, DataDefinition, Report
    from evidently.core.datasets import BinaryClassification
    from evidently.presets import ClassificationPreset, DataSummaryPreset

    current_labeled = current[current["true_label"].notna()].copy()

    if current_labeled.empty:
        logger.warning("classification: no true_label in current data")
        return "<html><body><h1>No labeled data</h1><p>Update predictions with true_label first.</p></body></html>"

    if reference.empty:
        logger.warning("classification: no reference - using DataSummaryPreset")
        report = Report(metrics=[DataSummaryPreset()])
        snapshot = report.run(current_data=current_labeled[["prediction"]])
        return snapshot.get_html_str(as_iframe=False)

    feature_cols = [c for c in FEATURE_COLUMNS if c in reference.columns and c in current_labeled.columns]
    cols = feature_cols + ["target", "prediction"]

    ref = reference[cols].copy()
    cur = current_labeled[feature_cols].copy()
    cur["target"] = current_labeled["true_label"].astype(int).values
    cur["prediction"] = current_labeled["prediction"].astype(int).values

    ref["target"] = ref["target"].astype(int)
    ref["prediction"] = ref["prediction"].astype(int)

    definition = DataDefinition(
        classification=[
            BinaryClassification(
                target="target",
                prediction_labels="prediction",
            )
        ]
    )

    ref_dataset = Dataset.from_pandas(ref, data_definition=definition)
    cur_dataset = Dataset.from_pandas(cur, data_definition=definition)

    report = Report(metrics=[ClassificationPreset()])
    snapshot = report.run(
        current_data=cur_dataset,
        reference_data=ref_dataset,
    )
    return snapshot.get_html_str(as_iframe=False)


def generate_data_quality_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> str:
    """Data Quality Report - missing values, outliers, statistics."""
    from evidently import Report
    from evidently.presets import DataSummaryPreset

    feature_cols = [c for c in FEATURE_COLUMNS if c in current.columns]

    run_kwargs: dict = {"current_data": current[feature_cols]}
    if not reference.empty:
        cols = [c for c in feature_cols if c in reference.columns]
        run_kwargs["reference_data"] = reference[cols]

    report = Report(metrics=[DataSummaryPreset()])
    snapshot = report.run(**run_kwargs)
    return snapshot.get_html_str(as_iframe=False)


REPORT_GENERATORS: dict[str, object] = {
    "data_drift": generate_drift_report,
    "target_drift": generate_target_drift_report,
    "classification": generate_classification_report,
    "data_quality": generate_data_quality_report,
}


def generate_all_reports(
    session: Optional[Session] = None,
    save_dir: Optional[Path] = None,
) -> dict[str, str]:
    """Generate all monitoring reports and save as HTML.

    Args:
        session: SQLAlchemy session (creates new if None)
        save_dir: Directory for HTML files (default: backend/reports/)

    Returns:
        Dict of {report_name: html_string}

    Raises:
        ValueError: If current data is empty
    """
    own_session = session is None
    if own_session:
        session = SessionLocal()

    try:
        reference = get_reference_data(session)
        current = get_current_data(session)

        if reference.empty:
            logger.warning("Reference dataset is empty — using reference-free mode")

        if current.empty:
            raise ValueError("Current dataset is empty — send inference requests first")

        logger.info(
            "Generating reports: reference=%d rows, current=%d rows",
            len(reference),
            len(current),
        )

        save_dir = save_dir or REPORTS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, str] = {}
        for name, generator in REPORT_GENERATORS.items():
            logger.info("Generating %s report…", name)
            try:
                html = generator(reference, current)
                results[name] = html

                filepath = save_dir / f"{name}_report.html"
                filepath.write_text(html, encoding="utf-8")
                logger.info("  → saved %s", filepath)
            except Exception:
                logger.exception("Failed to generate %s report", name)
                results[name] = (
                    f"<html><body>"
                    f"<h1>Error generating {name} report</h1>"
                    f"<p>Check server logs for details.</p>"
                    f"</body></html>"
                )

        return results

    finally:
        if own_session:
            session.close()