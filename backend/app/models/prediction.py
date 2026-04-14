"""ORM models for prediction logs and raw inference input payloads."""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Prediction(Base):
    """Stores a single prediction output from either training or inference."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    customer_feature_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("customer_features.id"),
        nullable=True,
    )

    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    prediction_proba: Mapped[float] = mapped_column(Float, nullable=False)
    true_label: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    customer_feature = relationship("CustomerFeature", backref="predictions")
    inference_input = relationship(
        "InferenceInput", back_populates="prediction", uselist=False
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction(id={self.id}, pred={self.prediction}, "
            f"proba={self.prediction_proba:.4f}, source={self.source!r})>"
        )


class InferenceInput(Base):
    """Stores the raw JSON payload associated with an inference prediction."""

    __tablename__ = "inference_inputs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("predictions.id"), unique=True, nullable=False
    )
    year_birth: Mapped[int] = mapped_column(Integer, nullable=False)
    education: Mapped[str] = mapped_column(String(50), nullable=False)
    marital_status: Mapped[str] = mapped_column(String(50), nullable=False)
    income: Mapped[float] = mapped_column(Float, nullable=False)
    kidhome: Mapped[int] = mapped_column(Integer, nullable=False)
    teenhome: Mapped[int] = mapped_column(Integer, nullable=False)
    dt_customer: Mapped[str] = mapped_column(String(20), nullable=False)
    recency: Mapped[int] = mapped_column(Integer, nullable=False)
    mnt_wines: Mapped[int] = mapped_column(Integer, nullable=False)
    mnt_fruits: Mapped[int] = mapped_column(Integer, nullable=False)
    mnt_meat_products: Mapped[int] = mapped_column(Integer, nullable=False)
    mnt_fish_products: Mapped[int] = mapped_column(Integer, nullable=False)
    mnt_sweet_products: Mapped[int] = mapped_column(Integer, nullable=False)
    mnt_gold_prods: Mapped[int] = mapped_column(Integer, nullable=False)
    num_deals_purchases: Mapped[int] = mapped_column(Integer, nullable=False)
    num_web_purchases: Mapped[int] = mapped_column(Integer, nullable=False)
    num_catalog_purchases: Mapped[int] = mapped_column(Integer, nullable=False)
    num_store_purchases: Mapped[int] = mapped_column(Integer, nullable=False)
    num_web_visits_month: Mapped[int] = mapped_column(Integer, nullable=False)
    complain: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    prediction = relationship("Prediction", back_populates="inference_input")

    def __repr__(self) -> str:
        return f"<InferenceInput(id={self.id}, prediction_id={self.prediction_id})>"