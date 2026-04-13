"""ORM model for persisting training run results."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class TrainingResult(Base):
    """Stores evaluation metrics for each completed training run."""

    __tablename__ = "training_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    train_size: Mapped[int] = mapped_column(Integer, nullable=False)
    test_size: Mapped[int] = mapped_column(Integer, nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    balanced_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    precision: Mapped[float] = mapped_column(Float, nullable=False)
    recall: Mapped[float] = mapped_column(Float, nullable=False)
    f1_score: Mapped[float] = mapped_column(Float, nullable=False)
    roc_auc: Mapped[float] = mapped_column(Float, nullable=False)
    pr_auc: Mapped[float] = mapped_column(Float, nullable=False)
    mcc: Mapped[float] = mapped_column(Float, nullable=False)
    true_negatives: Mapped[int] = mapped_column(Integer, nullable=False)
    false_positives: Mapped[int] = mapped_column(Integer, nullable=False)
    false_negatives: Mapped[int] = mapped_column(Integer, nullable=False)
    true_positives: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<TrainingResult(id={self.id}, version={self.model_version}, "
            f"pr_auc={self.pr_auc:.4f})>"
        )