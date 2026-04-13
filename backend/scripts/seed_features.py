import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.database import SessionLocal
from app.ml.pipeline import clean_raw_data, engineer_features
from app.models.customer import Customer
from app.models.customer_feature import CustomerFeature
from app.models.lookup import EducationLevel, MaritalStatus


def _load_raw(session) -> pd.DataFrame:
    """Read raw customers with education/marital_status names."""
    rows = (
        session.query(
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
            Customer.complain.label("Complain"),
            Customer.response.label("Response"),
            Customer.timestamp.label("timestamp"),
        )
        .join(EducationLevel, Customer.education_level_id == EducationLevel.id)
        .join(MaritalStatus, Customer.marital_status_id == MaritalStatus.id)
        .all()
    )

    df = pd.DataFrame(rows)
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"]).dt.strftime("%m/%d/%Y")
    df["Complain"] = df["Complain"].astype(int)
    df["Response"] = df["Response"].astype(int)
    return df


def seed_features() -> None:
    print("Seeding customer_features …")

    with SessionLocal() as session:
        df_raw = _load_raw(session)
        print(f"  Raw rows: {len(df_raw)}")

        # keep Id for FK mapping
        ids = df_raw["Id"].copy()

        # clean + feature engineer
        df_clean = clean_raw_data(df_raw)
        remaining_ids = df_clean["Id"].tolist()

        df_fe = engineer_features(df_clean)
        df_fe["customer_id"] = remaining_ids
        df_fe["timestamp"] = df_clean["timestamp"].tolist()

        print(f"  After clean + FE: {len(df_fe)} rows")

        # Build row dicts
        records = []
        for _, row in df_fe.iterrows():
            records.append(
                {
                    "customer_id": int(row["customer_id"]),
                    "education": int(row["Education"]),
                    "income": float(row["Income"]),
                    "kidhome": int(row["Kidhome"]),
                    "teenhome": int(row["Teenhome"]),
                    "recency": int(row["Recency"]),
                    "num_store_purchases": int(row["NumStorePurchases"]),
                    "response": int(row["Response"]),
                    "age": int(row["Age"]),
                    "customer_tenure_days": int(row["Customer_Tenure_Days"]),
                    "mnt_wines_log": float(row["MntWines_log"]),
                    "mnt_meat_products_log": float(row["MntMeatProducts_log"]),
                    "mnt_fruits_log": float(row["MntFruits_log"]),
                    "mnt_fish_products_log": float(row["MntFishProducts_log"]),
                    "mnt_sweet_products_log": float(row["MntSweetProducts_log"]),
                    "mnt_gold_prods_log": float(row["MntGoldProds_log"]),
                    "num_catalog_purchases_log": float(row["NumCatalogPurchases_log"]),
                    "num_web_purchases_log": float(row["NumWebPurchases_log"]),
                    "total_mnt": float(row["TotalMnt"]),
                    "total_purchases": int(row["TotalPurchases"]),
                    "wine_ratio": float(row["WineRatio"]),
                    "meat_ratio": float(row["MeatRatio"]),
                    "premium_ratio": float(row["PremiumRatio"]),
                    "catalog_share": float(row["CatalogShare"]),
                    "is_alone": int(row["is_alone"]),
                    "timestamp": row["timestamp"],
                }
            )

        # Upsert (on conflict update)
        batch_size = 200
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            stmt = pg_insert(CustomerFeature).values(batch)
            update_dict = {
                c.name: c
                for c in stmt.excluded
                if c.name not in ("id", "customer_id")
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["customer_id"],
                set_=update_dict,
            )
            session.execute(stmt)

        session.commit()
        print(f"  Inserted/updated: {len(records)} customer_features rows")

    print("seed_features completed!")


if __name__ == "__main__":
    seed_features()
