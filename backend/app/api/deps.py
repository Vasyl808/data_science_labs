"""FastAPI dependency providers shared across all API endpoints."""

from collections.abc import Generator
from fastapi import Depends
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.services.training_service import TrainingService
from app.services.inference_service import InferenceService
from app.services.monitoring_service import MonitoringService


def get_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session and ensure it is closed after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_training_service(db: Session = Depends(get_db)) -> TrainingService:
    return TrainingService(db)


def get_inference_service(db: Session = Depends(get_db)) -> InferenceService:
    return InferenceService(db)


def get_monitoring_service(db: Session = Depends(get_db)) -> MonitoringService:
    return MonitoringService(db)