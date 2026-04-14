"""API endpoints for model training operations."""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.training import TrainResponse, TrainingResultItem, FeatureImportanceResponse
from app.services import training_service

router = APIRouter()


@router.post(
    "/train-model",
    response_model=TrainResponse,
    summary="Train the ML model",
    description=(
        "Reads raw customer data from the database, cleans it, performs a stratified "
        "80/20 split, applies feature engineering, trains a pipeline via ModelFactory, "
        "evaluates on the test set, saves the artefact to disk, persists a TrainingResult "
        "row, and bulk-inserts training predictions."
    ),
)
def train_model(db: Session = Depends(get_db)) -> TrainResponse:
    """Trigger a full model training run."""
    result = training_service.train_model(db)
    return TrainResponse(**result)


@router.get(
    "/training-results",
    response_model=list[TrainingResultItem],
    summary="List training results",
    description=(
        "Returns the most recent training runs ordered by creation time descending. "
        "Use the ``limit`` query parameter to control how many rows are returned."
    ),
)
def list_training_results(
    limit: int = Query(20, ge=1, le=200, description="Maximum number of results to return"),
    db: Session = Depends(get_db),
) -> list[TrainingResultItem]:
    """Fetch persisted training result rows from the database."""
    return training_service.list_training_results(db, limit=limit)


@router.get(
    "/feature-importance",
    response_model=FeatureImportanceResponse,
    summary="Get feature importance",
    description=(
        "Returns feature importance using all available methods for the current model. "
        "Linear models (LogisticRegression) return coefficient-based importance. "
        "Tree-based models (RandomForest, XGBoost) return feature_importances_."
    ),
)
def get_feature_importance() -> FeatureImportanceResponse:
    """Get feature importance from the current trained model."""
    try:
        result = training_service.get_feature_importance()
        return FeatureImportanceResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))