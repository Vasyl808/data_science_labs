"""
scripts/generate_inference_data.py
-----------------------------------
Generates synthetic inference requests by sampling from the original
training data with slight perturbations (to simulate data drift).

This ensures there is enough data in ``inference_inputs`` + ``predictions``
tables for meaningful monitoring reports.

Usage:
    cd backend
    python scripts/generate_inference_data.py [--count 50]
"""

import argparse
import os
import sys
import logging
import random

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


EDUCATION_VALUES = ["Basic", "2n Cycle", "Graduation", "Master", "PhD"]
MARITAL_VALUES = ["Single", "Married", "Divorced", "Together", "Widow"]


def _perturb_value(value, dtype: str, drift_factor: float = 0.15):
    """Apply a small random perturbation to simulate data drift."""
    if dtype == "int":
        noise = int(value * drift_factor * random.uniform(-1, 1))
        return max(0, int(value) + noise)
    elif dtype == "float":
        noise = value * drift_factor * random.uniform(-1, 1)
        return max(0.0, value + noise)
    return value


def generate_inference_records(n: int = 50) -> list[dict]:
    """Sample from the training CSV and create perturbed inference payloads."""
    from app.database import SessionLocal
    from app.models.customer import Customer
    from app.models.lookup import EducationLevel, MaritalStatus

    session = SessionLocal()
    try:
        # Load raw customer data
        customers = (
            session.query(
                Customer.year_birth,
                EducationLevel.name.label("education"),
                MaritalStatus.name.label("marital_status"),
                Customer.income,
                Customer.kidhome,
                Customer.teenhome,
                Customer.dt_customer,
                Customer.recency,
                Customer.mnt_wines,
                Customer.mnt_fruits,
                Customer.mnt_meat_products,
                Customer.mnt_fish_products,
                Customer.mnt_sweet_products,
                Customer.mnt_gold_prods,
                Customer.num_deals_purchases,
                Customer.num_web_purchases,
                Customer.num_catalog_purchases,
                Customer.num_store_purchases,
                Customer.num_web_visits_month,
                Customer.complain,
            )
            .join(EducationLevel, Customer.education_level_id == EducationLevel.id)
            .join(MaritalStatus, Customer.marital_status_id == MaritalStatus.id)
            .all()
        )

        if not customers:
            logger.error("No customers found in database. Run seed_data.py first.")
            return []

        df = pd.DataFrame(customers)
        logger.info("Loaded %d customers as inference source", len(df))

        # Sample n rows (with replacement if n > len)
        sampled = df.sample(n=n, replace=True, random_state=None).reset_index(drop=True)

        records = []
        for _, row in sampled.iterrows():
            record = {
                "year_birth": _perturb_value(row["year_birth"], "int", 0.02),
                "education": random.choice(EDUCATION_VALUES),
                "marital_status": random.choice(MARITAL_VALUES),
                "income": _perturb_value(float(row["income"]) if row["income"] else 40000.0, "float", 0.25),
                "kidhome": random.choice([0, 0, 0, 1, 1, 2]),
                "teenhome": random.choice([0, 0, 0, 1, 1, 2]),
                "dt_customer": row["dt_customer"].strftime("%Y-%m-%d") if row["dt_customer"] else "2013-01-01",
                "recency": _perturb_value(row["recency"], "int", 0.3),
                "mnt_wines": _perturb_value(row["mnt_wines"], "int", 0.3),
                "mnt_fruits": _perturb_value(row["mnt_fruits"], "int", 0.3),
                "mnt_meat_products": _perturb_value(row["mnt_meat_products"], "int", 0.3),
                "mnt_fish_products": _perturb_value(row["mnt_fish_products"], "int", 0.3),
                "mnt_sweet_products": _perturb_value(row["mnt_sweet_products"], "int", 0.3),
                "mnt_gold_prods": _perturb_value(row["mnt_gold_prods"], "int", 0.3),
                "num_deals_purchases": _perturb_value(row["num_deals_purchases"], "int", 0.2),
                "num_web_purchases": _perturb_value(row["num_web_purchases"], "int", 0.2),
                "num_catalog_purchases": _perturb_value(row["num_catalog_purchases"], "int", 0.2),
                "num_store_purchases": _perturb_value(row["num_store_purchases"], "int", 0.2),
                "num_web_visits_month": _perturb_value(row["num_web_visits_month"], "int", 0.2),
                "complain": int(row["complain"]) if row["complain"] else 0,
            }
            records.append(record)

        return records

    finally:
        session.close()


def send_to_api(records: list[dict]) -> None:
    """Send inference records directly through the service layer (no HTTP)."""
    from app.database import SessionLocal
    from app.schemas.inference import PredictRequest
    from app.services.inference_service import InferenceService

    session = SessionLocal()
    success = 0
    errors = 0

    try:
        service = InferenceService(session)
        for i, record in enumerate(records):
            try:
                request = PredictRequest(**record)
                result = service.predict(request)
                success += 1
                if (i + 1) % 10 == 0:
                    logger.info(
                        "  Progress: %d/%d (pred=%d, proba=%.4f)",
                        i + 1, len(records),
                        result["prediction"], result["prediction_proba"],
                    )
            except Exception as exc:
                errors += 1
                if errors <= 3:
                    logger.warning("  Record %d failed: %s", i, exc)

        logger.info("Done: %d success, %d errors out of %d total", success, errors, len(records))
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic inference data for monitoring"
    )
    parser.add_argument(
        "--count", type=int, default=50,
        help="Number of synthetic inference records to generate (default: 50)",
    )
    args = parser.parse_args()

    logger.info("Generating %d synthetic inference records…", args.count)
    records = generate_inference_records(n=args.count)

    if not records:
        logger.error("No records generated. Exiting.")
        sys.exit(1)

    logger.info("Sending %d records through inference pipeline…", len(records))
    send_to_api(records)
    logger.info("Synthetic inference data generation complete!")


if __name__ == "__main__":
    main()
