"""
DAG для Лабораторной работы №2.
Оркестрирует процесс обучения нескольких моделей, их сравнение,
выбор лучшей и регистрацию в MLflow Model Registry с алиасом @champion.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow

from lab2.scripts.data_loader import download_churn_data
from lab2.scripts.data_preprocessor import preprocess_churn_data
from lab2.scripts.model_trainer import ModelTrainer

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("lab2_model_comparison")

default_args = {
    'owner': 'student',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def task_prepare_data() -> None:
    """Шаг 1: Загрузка и подготовка данных для обучения."""
    download_churn_data()
    train_path, test_path = preprocess_churn_data()
    logging.info(f"Данные успешно подготовлены: {train_path}, {test_path}")

def task_train_and_compare() -> dict:
    """Шаг 2: Обучение кандидатов, расчет метрик и выбор победителя."""
    trainer = ModelTrainer(experiment_name="lab2_model_comparison")
    results, best_model_name, best_run_id = trainer.train_and_evaluate()
    
    return {
        'best_model_name': best_model_name,
        'best_run_id': best_run_id,
        'results': results
    }

def task_register_model(**context) -> None:
    """Шаг 3: Регистрация лучшей модели в реестре и присвоение алиаса @champion."""
    data = context['ti'].xcom_pull(task_ids='train_and_compare_models')
    best_model_name = data['best_model_name']
    best_run_id = data['best_run_id']
    
    trainer = ModelTrainer()
    version = trainer.register_best_model(
        model_name=best_model_name,
        run_id=best_run_id,
        registered_model_name="bank_churn_model"
    )
    logging.info(f"Модель {best_model_name} (версия {version}) зарегистрирована как @champion.")

def task_generate_report(**context) -> None:
    """Шаг 4: Формирование сводного отчета о сравнении моделей."""
    data = context['ti'].xcom_pull(task_ids='train_and_compare_models')
    
    report_path = "/opt/airflow/data/final_model_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "winner_model": data['best_model_name'],
            "models_comparison": data['results']
        }, f, indent=4)
        
    with mlflow.start_run(run_name="00_final_comparison_summary", nested=True):
        mlflow.log_artifact(report_path)
        mlflow.set_tag("best_model", data['best_model_name'])
        
    logging.info("Сводный отчет успешно сохранен в артефакты MLflow.")

with DAG(
    dag_id='churn_model_training',
    default_args=default_args,
    description='Lab 2: Model comparison and MLflow Registry management',
    schedule=None,
    catchup=False,
    tags=['mlops', 'lab2', 'model_comparison']
) as dag:

    prepare_task = PythonOperator(task_id='prepare_data', python_callable=task_prepare_data)
    train_task = PythonOperator(task_id='train_and_compare_models', python_callable=task_train_and_compare)
    register_task = PythonOperator(task_id='register_best_model', python_callable=task_register_model)
    report_task = PythonOperator(task_id='generate_comparison_report', python_callable=task_generate_report)

    prepare_task >> train_task >> register_task >> report_task