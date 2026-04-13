"""ORM model for engineered customer features (one-to-one with customers)."""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.utils.random_data_generation import get_random_2025_timestamp


class CustomerFeature(Base):
    """Stores pre-computed, model-ready feature values derived from a Customer record."""

    __tablename__ = "customer_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("customers.id"), unique=True, nullable=False
    )
    education: Mapped[int] = mapped_column(Integer, nullable=False)
    income: Mapped[float] = mapped_column(Float, nullable=False)
    kidhome: Mapped[int] = mapped_column(Integer, nullable=False)
    teenhome: Mapped[int] = mapped_column(Integer, nullable=False)
    recency: Mapped[int] = mapped_column(Integer, nullable=False)
    num_store_purchases: Mapped[int] = mapped_column(Integer, nullable=False)
    response: Mapped[int] = mapped_column(Integer, nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    customer_tenure_days: Mapped[int] = mapped_column(Integer, nullable=False)
    mnt_wines_log: Mapped[float] = mapped_column(Float, nullable=False)
    mnt_meat_products_log: Mapped[float] = mapped_column(Float, nullable=False)
    mnt_fruits_log: Mapped[float] = mapped_column(Float, nullable=False)
    mnt_fish_products_log: Mapped[float] = mapped_column(Float, nullable=False)
    mnt_sweet_products_log: Mapped[float] = mapped_column(Float, nullable=False)
    mnt_gold_prods_log: Mapped[float] = mapped_column(Float, nullable=False)
    num_catalog_purchases_log: Mapped[float] = mapped_column(Float, nullable=False)
    num_web_purchases_log: Mapped[float] = mapped_column(Float, nullable=False)
    total_mnt: Mapped[float] = mapped_column(Float, nullable=False)
    total_purchases: Mapped[int] = mapped_column(Integer, nullable=False)
    wine_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    meat_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    premium_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    catalog_share: Mapped[float] = mapped_column(Float, nullable=False)
    is_alone: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=get_random_2025_timestamp, nullable=False
    )

    customer = relationship("Customer", backref="feature", uselist=False)


    def __repr__(self) -> str:
        return f"<CustomerFeature(id={self.id}, customer_id={self.customer_id})>"