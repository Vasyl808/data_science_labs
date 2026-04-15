"""Repository for Prediction and InferenceInput database operations."""

from typing import Any

from sqlalchemy.orm import Session

from app.models.prediction import InferenceInput, Prediction


class PredictionRepository:
    """Repository for Prediction and InferenceInput database operations."""

    def __init__(self, db: Session):
        self.db = db

    def add(self, prediction: Prediction) -> Prediction:
        """Add a prediction to the session."""
        self.db.add(prediction)
        return prediction

    def add_inference_input(self, inference_input: InferenceInput) -> InferenceInput:
        """Add an inference input to the session."""
        self.db.add(inference_input)
        return inference_input

    def bulk_save(self, predictions: list[Prediction]) -> None:
        """Bulk save predictions to the database."""
        self.db.bulk_save_objects(predictions)

    def get_by_id(self, prediction_id: int) -> Prediction | None:
        """Get a prediction by ID."""
        return self.db.query(Prediction).filter(Prediction.id == prediction_id).first()

    def update_true_label(self, prediction_id: int, true_label: int) -> bool:
        """Update the true label of a prediction.

        Returns True if updated, False if not found.
        """
        prediction = self.get_by_id(prediction_id)
        if prediction is None:
            return False

        prediction.true_label = true_label
        return True

    def get_inference_data(self) -> list[Any]:
        """Query inference inputs joined with predictions for current data.

        Used for monitoring reports.
        """
        return (
            self.db.query(
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
