"""Pydantic schemas for inference request and response payloads."""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
from datetime import datetime


class EducationEnum(str, Enum):
    basic = 'Basic'
    second_cycle = "2n Cycle"
    graduation = "Graduation"
    master = 'Master'
    phd = "PhD"


class MaritalStatusEnum(str, Enum):
    single = "Single"
    married = "Married"
    divorced = "Divorced"
    together = 'Together'
    widow = 'Widow'


class PredictRequest(BaseModel):
    """Raw customer data required to generate a campaign-response prediction."""
    model_config = ConfigDict(extra='forbid')

    year_birth: int = Field(..., ge=1900, le=2010, examples=[1970], description="Customer birth year")
    education: EducationEnum = Field(
        ..., 
        example="Graduation", 
        description="Education level. Allowed values: 'Basic', '2n Cycle', 'Graduation', 'Master', 'PhD'"
    )
    marital_status: MaritalStatusEnum = Field(
        ..., 
        example="Single", 
        description="Marital status. Allowed values: 'Single', 'Married', 'Divorced', 'Together', 'Widow'"
    )
    income: float = Field(..., ge=0.0, examples=[58138.0], description="Annual household income")
    kidhome: int = Field(..., ge=0, le=10, examples=[0], description="Number of small children")
    teenhome: int = Field(..., ge=0, le=10, examples=[0], description="Number of teenagers")
    dt_customer: str = Field(
        ..., min_length=8, max_length=10, examples=["2014-06-16"], description="Registration date (YYYY-MM-DD)"
    )
    recency: int = Field(..., ge=0, examples=[58], description="Days since last purchase")
    mnt_wines: int = Field(..., ge=0, examples=[635])
    mnt_fruits: int = Field(..., ge=0, examples=[88])
    mnt_meat_products: int = Field(..., ge=0, examples=[546])
    mnt_fish_products: int = Field(..., ge=0, examples=[172])
    mnt_sweet_products: int = Field(..., ge=0, examples=[88])
    mnt_gold_prods: int = Field(..., ge=0, examples=[88])
    num_deals_purchases: int = Field(..., ge=0, examples=[3])
    num_web_purchases: int = Field(..., ge=0, examples=[8])
    num_catalog_purchases: int = Field(..., ge=0, examples=[10])
    num_store_purchases: int = Field(..., ge=0, examples=[4])
    num_web_visits_month: int = Field(..., ge=0, examples=[7])
    complain: int = Field(
        ..., ge=0, le=1, examples=[0], description="Complained in last 2 years (0/1)"
    )

    @field_validator("dt_customer")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            pass
        try:
            datetime.strptime(v, "%m/%d/%Y")
            return v
        except ValueError:
            raise ValueError("dt_customer must be in format YYYY-MM-DD or MM/DD/YYYY")


class PredictResponse(BaseModel):
    """Response body returned after a successful prediction."""

    prediction_id: int
    prediction: int
    prediction_proba: float
    model_version: str


class UpdateTrueLabelRequest(BaseModel):
    """Request body for updating the true label of a prediction."""
    model_config = ConfigDict(extra='forbid')

    true_label: int = Field(..., ge=0, le=1, description="Actual outcome (0 or 1)")


class UpdateTrueLabelResponse(BaseModel):
    """Response body after updating a prediction's true label."""

    prediction_id: int
    true_label: int
    message: str