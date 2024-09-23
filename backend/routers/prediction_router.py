# app/routers/prediction_router.py
from fastapi import APIRouter, Depends
from controllers.prediction_controller import PredictionController
from utils.logger import get_model_usage_logs

router = APIRouter()


def get_controller():
    return PredictionController()


@router.get("/predict/lstm")
async def predict_lstm(
    days: int = 30, controller: PredictionController = Depends(get_controller)
):
    return controller.predict_lstm(days)


@router.get("/predict/rsi")
async def predict_rsi(
    days: int = 30, controller: PredictionController = Depends(get_controller)
):
    return controller.predict_rsi(days)


@router.get("/model_usage_logs")
async def get_usage_logs():
    return get_model_usage_logs()
