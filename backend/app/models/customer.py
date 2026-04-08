from datetime import date, datetime
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

from app.utils.random_data_generation import get_random_2025_timestamp


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)

    year_birth: Mapped[int] = mapped_column(Integer, nullable=False)
    income: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="Annual household income"
    )
    kidhome: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="Number of small children"
    )
    teenhome: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="Number of teenagers"
    )
    dt_customer: Mapped[Optional[date]] = mapped_column(
        Date, nullable=True, comment="Customer registration date"
    )

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

    num_deals_purchases: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="Purchases with discount"
    )
    num_web_purchases: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="Purchases through website"
    )
    num_catalog_purchases: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="Purchases through catalog"
    )
    num_store_purchases: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="Purchases in store"
    )
    num_web_visits_month: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="Website visits per month"
    )

    recency: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="Days since last purchase"
    )
    complain: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, comment="Complaint in the last 2 years"
    )
    response: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, comment="Accepted gold membership offer"
    )

    education_level: Mapped[Optional["EducationLevel"]] = relationship(
        back_populates="customers"
    )
    marital_status: Mapped[Optional["MaritalStatus"]] = relationship(
        back_populates="customers"
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime, 
        default=get_random_2025_timestamp,
        nullable=False,
        comment="Randomly generated timestamp within 2025"
    )

    def __repr__(self) -> str:
        return (
            f"<Customer("
            f"id={self.id}, "
            f"year_birth={self.year_birth}, "
            f"income={self.income}, "
            f"kidhome={self.kidhome}, "
            f"teenhome={self.teenhome}, "
            f"dt_customer={self.dt_customer}, "
            f"education_level_id={self.education_level_id}, "
            f"marital_status_id={self.marital_status_id}, "
            f"mnt_wines={self.mnt_wines}, "
            f"mnt_fruits={self.mnt_fruits}, "
            f"mnt_meat_products={self.mnt_meat_products}, "
            f"mnt_fish_products={self.mnt_fish_products}, "
            f"mnt_sweet_products={self.mnt_sweet_products}, "
            f"mnt_gold_prods={self.mnt_gold_prods}, "
            f"num_deals_purchases={self.num_deals_purchases}, "
            f"num_web_purchases={self.num_web_purchases}, "
            f"num_catalog_purchases={self.num_catalog_purchases}, "
            f"num_store_purchases={self.num_store_purchases}, "
            f"num_web_visits_month={self.num_web_visits_month}, "
            f"recency={self.recency}, "
            f"complain={self.complain}, "
            f"response={self.response}, "
            f"timestamp={self.timestamp}"
            f")>"
        )
