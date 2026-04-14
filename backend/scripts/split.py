import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
from app.ml.registry import save_model
from app.models.customer import Customer
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


def get_test_data_ids(db):
    df_raw = _load_raw_customers(db)
    df_clean = clean_raw_data(df_raw)

    df_train, df_test = train_test_split(
        df_clean,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
        stratify=df_clean["Response"],
    )

    true_labels = df_test["Response"].copy()

    drop_cols = ["Response", "Id"]
    X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])

    rename_map = {
        "Year_Birth": "year_birth", "Education": "education", "Marital_Status": "marital_status",
        "Income": "income", "Kidhome": "kidhome", "Teenhome": "teenhome", "Dt_Customer": "dt_customer",
        "Recency": "recency", "MntWines": "mnt_wines", "MntFruits": "mnt_fruits",
        "MntMeatProducts": "mnt_meat_products", "MntFishProducts": "mnt_fish_products",
        "MntSweetProducts": "mnt_sweet_products", "MntGoldProds": "mnt_gold_prods",
        "NumDealsPurchases": "num_deals_purchases", "NumWebPurchases": "num_web_purchases",
        "NumCatalogPurchases": "num_catalog_purchases", "NumStorePurchases": "num_store_purchases",
        "NumWebVisitsMonth": "num_web_visits_month", "Complain": "complain"
    }
    X_test = X_test.rename(columns=rename_map)

    if "dt_customer" in X_test.columns:
        X_test["dt_customer"] = pd.to_datetime(X_test["dt_customer"], unit="ms", errors="ignore")
        X_test["dt_customer"] = pd.to_datetime(X_test["dt_customer"]).dt.strftime('%Y-%m-%d')

    X_test["true_label"] = true_labels.values

    X_test.to_json("test_data.jsonl", orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    from app.database import SessionLocal
    
    with SessionLocal() as db:
        get_test_data_ids(db)
        print("Тестові дані успішно записано у test_data.jsonl")