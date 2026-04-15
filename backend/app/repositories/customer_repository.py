"""Repository for Customer database operations."""

from typing import Any

from sqlalchemy.orm import Session

from app.models.customer import Customer
from app.models.lookup import EducationLevel, MaritalStatus


class CustomerRepository:
    """Repository for Customer-related database operations."""

    def __init__(self, db: Session):
        self.db = db

    def get_raw_customers_with_lookups(self) -> list[Any]:
        """Query customers with joined education and marital status names.

        Returns a list of rows with CSV-like column aliases for ML pipeline compatibility.
        """
        return (
            self.db.query(
                Customer.id.label("Id"),
                Customer.year_birth.label("Year_Birth"),
                EducationLevel.name.label("Education"),
                MaritalStatus.name.label("Marital_Status"),
                Customer.income.label("Income"),
                Customer.kidhome.label("Kidhome"),
                Customer.teenhome.label("Teenhome"),
                Customer.dt_customer.label("Dt_Customer"),
                Customer.recency.label("Recency"),
                Customer.mnt_wines.label("MntWines"),
                Customer.mnt_fruits.label("MntFruits"),
                Customer.mnt_meat_products.label("MntMeatProducts"),
                Customer.mnt_fish_products.label("MntFishProducts"),
                Customer.mnt_sweet_products.label("MntSweetProducts"),
                Customer.mnt_gold_prods.label("MntGoldProds"),
                Customer.num_deals_purchases.label("NumDealsPurchases"),
                Customer.num_web_purchases.label("NumWebPurchases"),
                Customer.num_catalog_purchases.label("NumCatalogPurchases"),
                Customer.num_store_purchases.label("NumStorePurchases"),
                Customer.num_web_visits_month.label("NumWebVisitsMonth"),
                Customer.recency.label("Recency_dup"),
                Customer.complain.label("Complain"),
                Customer.response.label("Response"),
            )
            .join(EducationLevel, Customer.education_level_id == EducationLevel.id)
            .join(MaritalStatus, Customer.marital_status_id == MaritalStatus.id)
            .all()
        )
