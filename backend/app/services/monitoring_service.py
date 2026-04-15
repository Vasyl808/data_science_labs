from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.ml.pipeline import (
    ALONE_STATUSES,
    EDUCATION_ORDER,
    FEATURE_COLUMNS,
    LOG_COLS,
    MNT_COLS,
    SNAPSHOT_DATE_STR,
)
from app.repositories import CustomerFeatureRepository, PredictionRepository
from app.utils.report_generators import ReportGeneratorFactory

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


def _engineer_inference_features(df_raw: pd.DataFrame) -> pd.DataFrame:
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
    df["PremiumRatio"] = (df["MntWines"] + df["MntMeatProducts"]) / (df["TotalMnt"] + 1)
    df["CatalogShare"] = df["NumCatalogPurchases"] / (df["TotalPurchases"] + 1)

    for col in LOG_COLS:
        df[f"{col}_log"] = np.log1p(df[col])

    df["Education"] = df["Education"].map(EDUCATION_ORDER)
    df["is_alone"] = df["Marital_Status"].isin(ALONE_STATUSES).astype(int)

    return df


def _eval_result_to_html(eval_result) -> str:
    """Evidently 0.7 uses save_html; convert it back to a string for APIs."""
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        eval_result.save_html(str(tmp_path))
        return tmp_path.read_text(encoding="utf-8")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


class MonitoringService:
    def __init__(self, db: Session):
        self.db = db
        self.customer_feature_repo = CustomerFeatureRepository(db)
        self.prediction_repo = PredictionRepository(db)

    def generate_single_report(self, report_type: str) -> str:
        reference = self.get_reference_data()
        current = self.get_current_data()

        if reference.empty:
            logger.warning(
                "generate_single_report: reference is empty — running in reference-free mode"
            )

        if current.empty:
            raise ValueError(
                "Current dataset is empty — send inference requests or run generate_inference_data.py."
            )

        generator = ReportGeneratorFactory.get_generator(report_type)
        return generator.generate(reference, current)

    def _get_reference_data(self) -> pd.DataFrame:
        rows = self.customer_feature_repo.get_reference_data_with_predictions()

        if not rows:
            logger.warning("No reference data found with true_label")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df.rename(columns=_CF_TO_FEATURE, inplace=True)
        df.rename(columns={"true_label": "target"}, inplace=True)

        feature_and_target_cols = list(_CF_TO_FEATURE.values()) + ["target"]
        df = (
            df.sort_values("timestamp", ascending=False)
            .drop_duplicates(subset=feature_and_target_cols)
            .reset_index(drop=True)
        )

        logger.info("Reference data loaded: %d rows", len(df))
        return df

    def _get_current_data(self) -> pd.DataFrame:
        rows = self.prediction_repo.get_inference_data()

        if not rows:
            logger.warning("No inference data found")
            return pd.DataFrame()

        df_raw = pd.DataFrame(rows).reset_index(drop=True)
        df_fe = _engineer_inference_features(df_raw)

        result = df_fe[FEATURE_COLUMNS].copy()
        result["prediction"] = df_raw["prediction"]
        result["true_label"] = df_raw["true_label"]
        result["timestamp"] = df_raw["created_at"]

        logger.info(
            "Current data loaded: %d rows (%d with true_label)",
            len(result),
            result["true_label"].notna().sum(),
        )
        return result

    def generate_all_reports(self, save_dir: Optional[Path] = None) -> dict[str, str]:
        reference = self._get_reference_data()
        current = self._get_current_data()

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
        for name in ReportGeneratorFactory.get_available_reports():
            logger.info("Generating %s report…", name)
            try:
                generator = ReportGeneratorFactory.get_generator(name)
                html = generator.generate(reference, current)
                results[name] = html

                filepath = save_dir / f"{name}_report.html"
                filepath.write_text(html, encoding="utf-8")
                logger.info("  → saved %s", filepath)
            except Exception:
                logger.exception("Failed to generate %s report", name)
                results[name] = (
                    f"<html><body><h1>Error generating {name} report</h1>"
                    f"<p>Check server logs for details.</p></body></html>"
                )

        return results