"""Application entry-point — creates the FastAPI app and configures lifespan."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.v1.router import api_v1_router
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Ensure the models directory exists before accepting requests."""
    Path(settings.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    logging.getLogger(__name__).info(
        "Models directory ready: %s", settings.MODEL_DIR
    )
    yield


app = FastAPI(
    title="Superstore ML API",
    description=(
        "Machine-learning service for predicting customer campaign response. "
        "Trained with Logistic Regression (elasticnet) on feature-engineered "
        "Superstore data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_v1_router, prefix="/api/v1")