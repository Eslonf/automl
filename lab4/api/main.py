"""
Микросервис инференса на базе FastAPI.
Обеспечивает отдачу предсказаний, метрики Prometheus и горячую перезагрузку модели.
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field
import mlflow
import mlflow.pyfunc
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("inference_api")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

MODEL_NAME = "bank_churn_model"
model: Optional[mlflow.pyfunc.PyFuncModel] = None
model_version: Optional[str] = None

# Изолированный реестр Prometheus для предотвращения дублирования метрик
registry = CollectorRegistry()

PREDICTIONS_COUNT = Counter(
    "model_predictions_total",
    "Общее количество запросов на предсказание",
    ["status"],
    registry=registry
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Время выполнения предсказания",
    registry=registry
)

def load_production_model() -> bool:
    """Загрузка активной модели из MLflow Registry по алиасу @champion."""
    global model, model_version
    try:
        # Современный подход: загрузка по алиасу
        model_uri = f"models:/{MODEL_NAME}@champion"
        logger.info(f"Попытка загрузки модели: {model_uri}...")
        model = mlflow.pyfunc.load_model(model_uri)

        # Получение метаданных версии
        client = mlflow.tracking.MlflowClient()
        model_info = client.get_model_version_by_alias(MODEL_NAME, "champion")
        model_version = str(model_info.version)

        logger.info(f"✅ Модель {MODEL_NAME} (версия: {model_version}) успешно загружена!")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Ошибка загрузки модели {MODEL_NAME}: {e}")
        model, model_version = None, None
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом сервиса FastAPI."""
    logger.info("Инициализация сервиса инференса...")
    load_production_model()
    yield
    logger.info("Завершение работы сервиса...")

app = FastAPI(
    title="Bank Churn Prediction API",
    description="REST API микросервис для инференса модели оттока клиентов банка",
    version="2.0.0",
    lifespan=lifespan
)

class CustomerFeatures(BaseModel):
    """Схема валидации входных признаков клиента."""
    CreditScore: float = Field(619.0, description="Кредитный рейтинг")
    Geography: int = Field(0, description="Страна (0: France, 1: Germany, 2: Spain)")
    Gender: int = Field(0, description="Пол (0: Female, 1: Male)")
    Age: float = Field(42.0, description="Возраст")
    Tenure: float = Field(2.0, description="Лет с банком")
    Balance: float = Field(0.0, description="Баланс")
    NumOfProducts: float = Field(1.0, description="Кол-во продуктов")
    HasCrCard: float = Field(1.0, description="Наличие кредитки (1/0)")
    IsActiveMember: float = Field(1.0, description="Активность (1/0)")
    EstimatedSalary: float = Field(101348.88, description="Зарплата")

class PredictionResponse(BaseModel):
    """Схема ответа сервиса."""
    prediction: int = Field(..., description="Класс оттока (1 - уйдет, 0 - останется)")
    status: str = Field(..., description="Текстовый статус ('Churn' или 'Loyal')")
    model_name: str
    model_version: Optional[str]

@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    """Эндпоинт для получения предсказания модели."""
    global model, model_version

    if model is None:
        if not load_production_model():
            PREDICTIONS_COUNT.labels(status="model_unavailable").inc()
            raise HTTPException(status_code=503, detail="Модель не готова.")

    start_time = time.time()
    try:
        input_data = pd.DataFrame([features.model_dump()])
        raw_prediction = model.predict(input_data)
        pred_value = int(raw_prediction[0])

        PREDICTIONS_COUNT.labels(status="success").inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)

        return PredictionResponse(
            prediction=pred_value,
            status="Churn" if pred_value == 1 else "Loyal",
            model_name=MODEL_NAME,
            model_version=model_version
        )
    except Exception as e:
        PREDICTIONS_COUNT.labels(status="error").inc()
        logger.error(f"Ошибка инференса: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    """Проверка состояния сервиса."""
    is_ready = model is not None
    return {
        "status": "healthy" if is_ready else "waiting_for_model",
        "model_loaded": is_ready,
        "model_version": model_version
    }

@app.post("/reload")
def reload_model():
    """Горячая перезагрузка модели (вызывается из Airflow после переобучения)."""
    if load_production_model():
        return {"status": "reloaded", "model_version": model_version}
    raise HTTPException(status_code=500, detail="Не удалось перезагрузить модель.")

@app.get("/metrics")
def metrics():
    """Экспорт метрик для Prometheus."""
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)