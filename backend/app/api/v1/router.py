"""API v1 router — aggregates all endpoint sub-routers."""

from fastapi import APIRouter

from app.api.v1.endpoints import inference, training

api_v1_router = APIRouter()
api_v1_router.include_router(training.router, tags=["Training"])
api_v1_router.include_router(inference.router, tags=["Inference"])