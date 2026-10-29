"""
DAG для Лабораторной работы №4.
Оркестрирует мониторинг дрейфа данных, условное переобучение модели
и отправку Webhook-сигнала в FastAPI для горячей перезагрузки весов.
"""

import os
import logging
import requests
from datetime import datetime, timedelta
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator, BranchPythonOperator
import mlflow

from lab4.scripts.data_drift_detector import DataDriftMonitor
from lab4.scripts.model_retraining import ModelRetrainer

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

default_args = {
    'owner': 'student',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}


def task_generate_incoming_batch(**context) -> str:
    """Шаг 1: Симуляция поступления новых данных из production."""
    simulate_drift = context['params'].get('simulate_drift', True)
    monitor = DataDriftMonitor()
    return monitor.generate_current_data_with_drift(simulate_drift=simulate_drift)


def task_analyze_data_drift(**context) -> None:
    """Шаг 2: Анализ дрейфа с использованием Evidently AI."""
    batch_path = context['ti'].xcom_pull(task_ids='simulate_incoming_batch')
    monitor = DataDriftMonitor()
    drift_detected, drift_share, report_path = monitor.detect_drift(batch_path)

    context['ti'].xcom_push(key='drift_detected', value=drift_detected)


def branch_decision_step(**context) -> str:
    """Шаг 3: Ветвление пайплайна на основе наличия дрейфа."""
    drift_detected = context['ti'].xcom_pull(task_ids='analyze_data_drift', key='drift_detected')
    if drift_detected:
        logging.info("🚨 Обнаружен дрейф данных. Активация ветки переобучения.")
        return 'trigger_retraining'
    logging.info("✅ Дрейф отсутствует. Переобучение не требуется.")
    return 'no_retraining_needed'


def task_execute_retraining() -> None:
    """Шаг 4А: Переобучение модели и регистрация новой версии."""
    retrainer = ModelRetrainer()
    new_version, new_f1 = retrainer.retrain_model()
    logging.info(f"Модель успешно дообучена (v{new_version}, F1: {new_f1:.4f})")


def task_reload_api() -> None:
    """Шаг 5: Отправка сигнала в FastAPI для горячей перезагрузки модели."""
    try:
        response = requests.post("http://fastapi:8000/reload", timeout=10)
        response.raise_for_status()
        logging.info(f"API успешно перезагружено: {response.json()}")
    except Exception as e:
        logging.error(f"Ошибка при отправке webhook в API: {e}")
        raise


def task_skip_retraining() -> None:
    """Шаг 4Б: Заглушка при отсутствии дрейфа."""
    logging.info("Пайплайн завершен без изменений модели.")


with DAG(
    dag_id='mlops_monitoring_pipeline',
    default_args=default_args,
    description='Lab 4: End-to-End MLOps Pipeline with Drift Detection',
    schedule=None,
    catchup=False,
    tags=['mlops', 'lab4', 'evidently', 'monitoring'],
    params={
        "simulate_drift": Param(
            default=True,
            type="boolean",
            description="Включить искусственный дрейф признаков (True -> переобучение, False -> пропуск)"
        )
    }
) as dag:

    t_batch = PythonOperator(task_id='simulate_incoming_batch', python_callable=task_generate_incoming_batch)
    t_drift = PythonOperator(task_id='analyze_data_drift', python_callable=task_analyze_data_drift)
    t_branch = BranchPythonOperator(task_id='evaluate_drift_decision', python_callable=branch_decision_step)
    t_retrain = PythonOperator(task_id='trigger_retraining', python_callable=task_execute_retraining)
    t_reload_api = PythonOperator(task_id='reload_fastapi_model', python_callable=task_reload_api)
    t_skip = PythonOperator(task_id='no_retraining_needed', python_callable=task_skip_retraining)

    t_batch >> t_drift >> t_branch
    t_branch >> t_retrain >> t_reload_api
    t_branch >> t_skip