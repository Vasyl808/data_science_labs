"""Repository for CustomerFeature database operations."""

from typing import Any

from sqlalchemy.orm import Session

from app.models.customer_feature import CustomerFeature


class CustomerFeatureRepository:
    """Repository for CustomerFeature-related database operations."""

    def __init__(self, db: Session):
        self.db = db

    def get_by_customer_ids(self, customer_ids: list[int]) -> list[Any]:
        """Query customer features by customer IDs.

        Returns rows with id, customer_id, and response for prediction mapping.
        """
        return (
            self.db.query(CustomerFeature.id, CustomerFeature.customer_id, CustomerFeature.response)
            .filter(CustomerFeature.customer_id.in_(customer_ids))
            .all()
        )

    def get_reference_data_with_predictions(self) -> list[Any]:
        """Query customer features joined with predictions for reference data.

        Used for monitoring reports - returns training data with true labels.
        """
        from app.models.prediction import Prediction

        return (
            self.db.query(
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
            .join(Prediction, CustomerFeature.id == Prediction.customer_feature_id)
            .filter(Prediction.source == "train")
            .filter(Prediction.true_label.isnot(None))
            .all()
        )
