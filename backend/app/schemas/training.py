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