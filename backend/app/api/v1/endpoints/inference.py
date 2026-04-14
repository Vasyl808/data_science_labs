from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api.deps import get_db
from app.schemas.inference import PredictRequest, PredictResponse, UpdateTrueLabelRequest, UpdateTrueLabelResponse
from app.services import inference_service

router = APIRouter()


@router.post('/predict', response_model=PredictResponse, summary='Predict customer response', description='Accepts raw customer data (same columns as original CSV, without Id/Response), applies feature engineering, loads the latest trained model, predicts, and saves both the prediction and the raw input payload to the database.')
def predict(request: PredictRequest, db: Session=Depends(get_db)) -> PredictResponse:
    result = inference_service.predict(request, db)
    return PredictResponse(**result)


@router.patch(
    '/predictions/{prediction_id}/true-label',
    response_model=UpdateTrueLabelResponse,
    summary='Update true label',
    description='Update the true label of a prediction after the actual outcome is known (e.g., after calling the client).',
)
def update_true_label(
    prediction_id: int,
    request: UpdateTrueLabelRequest,
    db: Session = Depends(get_db),
) -> UpdateTrueLabelResponse:
    result = inference_service.update_true_label(prediction_id, request.true_label, db)
    if result is None:
        raise HTTPException(status_code=404, detail=f'Prediction with id={prediction_id} not found')
    return UpdateTrueLabelResponse(**result)