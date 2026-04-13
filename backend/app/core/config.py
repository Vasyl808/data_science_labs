"""Application-wide settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralised configuration — values come from .env or OS environment."""

    DATABASE_URL: str

    MODEL_DIR: str = str(Path(__file__).resolve().parent.parent.parent / "models")
    MODEL_NAME_PREFIX: str = "logistic_regression_weighted"
    MODEL_TYPE: str = "logistic_regression"

    LR_SOLVER: str = "saga"
    LR_C: float = 18.06362747243349
    LR_PENALTY: str = "elasticnet"
    LR_L1_RATIO: float = 0.4401315633673214
    LR_MAX_ITER: int = 5000
    LR_CLASS_WEIGHT: str = "balanced"

    TEST_SIZE: float = 0.20
    RANDOM_STATE: int = 42

    SNAPSHOT_YEAR: int = 2014
    SNAPSHOT_DATE: str = "2014-06-30"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
