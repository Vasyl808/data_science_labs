"""Feature engineering pipeline and model factory for the ML layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

if TYPE_CHECKING:
    from app.core.config import Settings


SNAPSHOT_DATE_STR = "2014-12-07"

EDUCATION_ORDER: dict[str, int] = {
    "Basic": 0,
    "2n Cycle": 1,
    "Graduation": 2,
    "Master": 3,
    "PhD": 4,
}

ALONE_STATUSES: frozenset[str] = frozenset({"Single", "Widow", "Divorced"})

MNT_COLS: list[str] = [
    "MntWines",
    "MntMeatProducts",
    "MntFruits",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]

LOG_COLS: list[str] = [
    "MntWines",
    "MntMeatProducts",
    "MntFruits",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumCatalogPurchases",
    "NumWebPurchases",
]

FEATURE_COLUMNS: list[str] = [
    "Education",
    "Income",
    "Kidhome",
    "Teenhome",
    "Recency",
    "NumStorePurchases",
    "Age",
    "Customer_Tenure_Days",
    "MntWines_log",
    "MntMeatProducts_log",
    "MntFruits_log",
    "MntFishProducts_log",
    "MntSweetProducts_log",
    "MntGoldProds_log",
    "NumCatalogPurchases_log",
    "NumWebPurchases_log",
    "TotalMnt",
    "TotalPurchases",
    "WineRatio",
    "MeatRatio",
    "PremiumRatio",
    "CatalogShare",
    "is_alone",
]

TARGET_COLUMN: str = "Response"


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, NaNs, invalid statuses, and outliers.

    Must be called only on the full training batch before splitting.
    """
    df = df.copy()

    df = df.drop_duplicates(subset=df.columns.difference(["Id"]))

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%m/%d/%Y")

    df.dropna(inplace=True)

    bad_statuses = ["Alone", "YOLO", "Absurd"]
    df = df[~df["Marital_Status"].isin(bad_statuses)]

    current_year = 2014
    age = current_year - df["Year_Birth"]
    q1, q3 = age.quantile(0.25), age.quantile(0.75)
    iqr = q3 - q1
    age_mask = (age >= q1 - 1.5 * iqr) & (age <= q3 + 1.5 * iqr)

    income_mask = df["Income"] != 666666.0

    return df[age_mask & income_mask].reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw columns into model-ready features.

    Works on both batch DataFrames and single-row DataFrames.
    Returns a DataFrame with FEATURE_COLUMNS (and Response if present).
    """
    df = df.copy()

    df["Dt_Customer"] = pd.to_datetime(
        df["Dt_Customer"], errors="coerce"
    )
    snapshot = pd.Timestamp(SNAPSHOT_DATE_STR)

    df["Age"] = 2014 - df["Year_Birth"]
    df["Customer_Tenure_Days"] = (snapshot - df["Dt_Customer"]).dt.days

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

    output_cols = list(FEATURE_COLUMNS)
    if TARGET_COLUMN in df.columns:
        output_cols.append(TARGET_COLUMN)

    return df[output_cols]


log_features = [
    "MntWines_log", "MntMeatProducts_log", "MntFruits_log",
    "MntFishProducts_log", "MntSweetProducts_log", "MntGoldProds_log",
    "NumCatalogPurchases_log", "NumWebPurchases_log",
]
raw_features = [
    "Income", "Recency", "Age", "Customer_Tenure_Days",
    "NumStorePurchases", "TotalMnt", "TotalPurchases",
    "WineRatio", "MeatRatio", "PremiumRatio", "CatalogShare",
]
passthrough_features = ["Education", "Kidhome", "Teenhome", "is_alone"]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("std",    StandardScaler(), log_features),
            ("robust", RobustScaler(),   raw_features),
            ("pass",   "passthrough",    passthrough_features),
        ],
        remainder="drop",
    )


def _build_logistic_regression(cfg: "Settings") -> Pipeline:
    """Build a ColumnTransformer → LogisticRegression pipeline from settings."""
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    solver=cfg.LR_SOLVER,
                    C=cfg.LR_C,
                    penalty=cfg.LR_PENALTY,
                    l1_ratio=cfg.LR_L1_RATIO,
                    max_iter=cfg.LR_MAX_ITER,
                    random_state=cfg.RANDOM_STATE,
                    class_weight=cfg.LR_CLASS_WEIGHT,
                ),
            ),
        ]
    )


_BUILDER_REGISTRY: dict[str, Callable[["Settings"], Pipeline]] = {
    "logistic_regression": _build_logistic_regression,
}


class ModelFactory:
    """Creates an sklearn Pipeline based on the active model type in Settings.

    To register a new model type, add an entry to ``_BUILDER_REGISTRY``
    and a corresponding builder function — no service or endpoint changes
    are required.
    """

    @classmethod
    def build(cls, cfg: "Settings") -> Pipeline:
        """Return an unfitted sklearn Pipeline for the configured model type.

        Parameters
        ----------
        cfg:
            Application settings instance providing model type and
            hyperparameter values.

        Raises
        ------
        ValueError
            If ``cfg.MODEL_TYPE`` is not registered in the builder registry.
        """
        builder = _BUILDER_REGISTRY.get(cfg.MODEL_TYPE)
        if builder is None:
            registered = list(_BUILDER_REGISTRY.keys())
            raise ValueError(
                f"Unknown model type: {cfg.MODEL_TYPE!r}. "
                f"Registered types: {registered}"
            )
        return builder(cfg)
