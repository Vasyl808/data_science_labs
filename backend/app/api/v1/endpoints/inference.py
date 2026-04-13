from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.api.deps import get_db
from app.schemas.inference import PredictRequest, PredictResponse
from app.services import inference_service

router = APIRouter()


@router.post('/predict', response_model=PredictResponse, summary='Predict customer response', description='Accepts raw customer data (same columns as original CSV, without Id/Response), applies feature engineering, loads the latest trained model, predicts, and saves both the prediction and the raw input payload to the database.')
def predict(request: PredictRequest, db: Session=Depends(get_db)) -> PredictResponse:
    result = inference_service.predict(request, db)
    return PredictResponse(**result)