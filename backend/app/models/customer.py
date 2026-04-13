"""ORM model for the raw customer record."""

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.utils.random_data_generation import get_random_2025_timestamp


class Customer(Base):
    """Stores raw customer data as imported from the original dataset."""

    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    year_birth: Mapped[int] = mapped_column(Integer, nullable=False)
    income: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    kidhome: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    teenhome: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    dt_customer: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    education_level_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("education_levels.id"), nullable=True
    )
    marital_status_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("marital_statuses.id"), nullable=True
    )
    mnt_wines: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    mnt_fruits: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    mnt_meat_products: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    mnt_fish_products: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    mnt_sweet_products: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    mnt_gold_prods: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    num_deals_purchases: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    num_web_purchases: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    num_catalog_purchases: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    num_store_purchases: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    num_web_visits_month: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    recency: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    complain: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    response: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    education_level: Mapped[Optional["EducationLevel"]] = relationship(
        back_populates="customers"
    )
    marital_status: Mapped[Optional["MaritalStatus"]] = relationship(
        back_populates="customers"
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=get_random_2025_timestamp, nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<Customer(id={self.id}, year_birth={self.year_birth}, "
            f"income={self.income})>"
        )