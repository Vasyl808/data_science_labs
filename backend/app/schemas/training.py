"""Pydantic schemas for training-related request and response payloads."""

from datetime import datetime

from pydantic import BaseModel


class TrainResponse(BaseModel):
    """Response body returned after a successful training run."""

    train_size: int
    test_size: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    mcc: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    model_version: str
    message: str


class TrainingResultItem(BaseModel):
    """Read schema for a single row from the training_results table."""

    id: int
    model_version: str
    train_size: int
    test_size: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    mcc: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    created_at: datetime

    model_config = {"from_attributes": True}


class FeatureImportanceItem(BaseModel):
    """Single feature with its importance score."""

    feature: str
    importance: float
    rank: int


class UnavailableMethod(BaseModel):
    """Method that is not available for the current model."""

    method: str
    reason: str


class FeatureImportanceResponse(BaseModel):
    """Feature importance from different methods."""

    model_version: str
    model_type: str
    available_methods: list[str]
    unavailable_methods: list[UnavailableMethod]
    coefficient_importance: list[FeatureImportanceItem] | None = None
    feature_importance: list[FeatureImportanceItem] | None = None

    model_config = {"exclude_none": True}