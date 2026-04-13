"""
scripts/seed_data.py
--------------------
Loads superstore_data.csv into the database.

Usage:
    python scripts/seed_data.py --csv superstore_data.csv
"""

import argparse
import os
import sys
from datetime import date, datetime

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import SessionLocal
from app.models.lookup import EducationLevel, MaritalStatus
from app.models.customer import Customer

from typing import Union

EDU_ORDER = {'Basic': 0, '2n Cycle': 1, 'Graduation': 2, 'Master': 3, 'PhD': 4}


def parse_date(val) -> Union[date, None]:
    if pd.isna(val):
        return None
    if isinstance(val, (datetime, pd.Timestamp)):
        return val.date()
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(val).strip(), fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def upsert_ordered_lookup(session, model, id_map: dict[str, int]) -> dict[str, int]:
    """Inserts lookup rows with explicit IDs. Returns {name: id}."""
    stmt = pg_insert(model).values([
        {"id": id_, "name": name} for name, id_ in id_map.items()
    ])
    stmt = stmt.on_conflict_do_update(
        index_elements=["id"],
        set_={"name": stmt.excluded.name},
    )
    session.execute(stmt)
    session.flush()
    return id_map


def upsert_lookup(session, model, values: list[str]) -> dict[str, int]:
    """Inserts new values into a lookup table, returns {name: id}."""
    existing = {row.name: row.id for row in session.query(model).all()}
    new_values = [v for v in values if v not in existing]

    if new_values:
        stmt = pg_insert(model).values([{"name": v} for v in new_values])
        stmt = stmt.on_conflict_do_nothing(index_elements=["name"])
        session.execute(stmt)
        session.flush()
        existing = {row.name: row.id for row in session.query(model).all()}

    return existing


def clear_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=df.columns.difference(["id"]))

    df['dt_customer'] = pd.to_datetime(df['dt_customer'], format='%m/%d/%Y')
    snapshot_date = df['dt_customer'].max() + pd.DateOffset(days=1)
    print(f'Snapshot date: {snapshot_date.date()}')

    df.dropna(inplace=True)
    print(f'Size after dropna: {df.shape}')

    statuses_to_drop = ['Alone', 'YOLO', 'Absurd']

    print(f'\nUnique Marital_Status before: {df["marital_status"].unique()}')
    print(f'Rows before: {len(df)}')

    df = df[~df['marital_status'].isin(statuses_to_drop)]

    print(f'Unique Marital_Status after: {df["marital_status"].unique()}')
    
    Q1, Q3 = df['year_birth'].quantile(0.25), df['year_birth'].quantile(0.75)
    IQR = Q3 - Q1
    age_mask = (df['year_birth'] >= Q1 - 1.5*IQR) & (df['year_birth'] <= Q3 + 1.5*IQR)

    income_mask = df['income'] != 666666.00

    df = df[age_mask & income_mask]
    
    print(f'Rows after: {len(df)}')

    return df


def seed(csv_path: str, batch_size: int = 200) -> None:
    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"Rows: {len(df)}")

    df = clear_data(df)

    with SessionLocal() as session:
        # 1. Lookup tables
        print("Lookup tables…")
        edu_map = upsert_ordered_lookup(session, EducationLevel, EDU_ORDER)

        marital_map = upsert_lookup(session, MaritalStatus,
                                    df["marital_status"].dropna().unique().tolist())
        session.commit()
        print(f"   Education    : {list(edu_map.keys())}")
        print(f"   Marital      : {list(marital_map.keys())}")

        # 2. Customers (all fields in one table)
        print("Customers…")
        rows = []
        for _, row in df.iterrows():
            rows.append({
                "id":                    int(row["id"]),
                "year_birth":            int(row["year_birth"]),
                "income":                float(row["income"]) if pd.notna(row.get("income")) else None,
                "kidhome":               int(row.get("kidhome", 0)),
                "teenhome":              int(row.get("teenhome", 0)),
                "dt_customer":           parse_date(row.get("dt_customer")),
                "education_level_id":    edu_map.get(row.get("education")),
                "marital_status_id":     marital_map.get(row.get("marital_status")),
                # spending
                "mnt_wines":             int(row.get("mntwines", 0)),
                "mnt_fruits":            int(row.get("mntfruits", 0)),
                "mnt_meat_products":     int(row.get("mntmeatproducts", 0)),
                "mnt_fish_products":     int(row.get("mntfishproducts", 0)),
                "mnt_sweet_products":    int(row.get("mntsweetproducts", 0)),
                "mnt_gold_prods":        int(row.get("mntgoldprods", 0)),
                # channels
                "num_deals_purchases":   int(row.get("numdealspurchases", 0)),
                "num_web_purchases":     int(row.get("numwebpurchases", 0)),
                "num_catalog_purchases": int(row.get("numcatalogpurchases", 0)),
                "num_store_purchases":   int(row.get("numstorepurchases", 0)),
                "num_web_visits_month":  int(row.get("numwebvisitsmonth", 0)),
                # campaign
                "recency":               int(row.get("recency", 0)),
                "complain":              bool(int(row.get("complain", 0))),
                "response":              bool(int(row.get("response", 0))),
            })

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            stmt = pg_insert(Customer).values(batch)
            update_dict = {
                c.name: c
                for c in stmt.excluded
                if c.name not in ["id"]
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                set_=update_dict
            )
            session.execute(stmt)
        session.commit()
        print(f"   Inserted: {len(rows)} customers")

    print("\n Seed completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="superstore_data.csv")
    args = parser.parse_args()
    seed(args.csv)
