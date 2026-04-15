from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import get_inference_service
from app.schemas.inference import PredictRequest, PredictResponse, UpdateTrueLabelRequest, UpdateTrueLabelResponse
from app.services.inference_service import InferenceService

router = APIRouter()


@router.post('/predict', response_model=PredictResponse, summary='Predict customer response', description='Accepts raw customer data (same columns as original CSV, without Id/Response), applies feature engineering, loads the latest trained model, predicts, and saves both the prediction and the raw input payload to the database.')
def predict(request: PredictRequest, service: InferenceService = Depends(get_inference_service)) -> PredictResponse:
    try:
        result = service.predict(request)
        return PredictResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch(
    '/predictions/{prediction_id}/true-label',
    response_model=UpdateTrueLabelResponse,
    summary='Update true label',
    description='Update the true label of a prediction after the actual outcome is known (e.g., after calling the client).',
)
def update_true_label(
    prediction_id: int,
    request: UpdateTrueLabelRequest,
    service: InferenceService = Depends(get_inference_service),
) -> UpdateTrueLabelResponse:
    result = service.update_true_label(prediction_id, request.true_label)
    if result is None:
        raise HTTPException(status_code=404, detail=f'Prediction with id={prediction_id} not found')
    return UpdateTrueLabelResponse(**result)