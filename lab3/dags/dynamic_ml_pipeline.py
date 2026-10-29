"""
DAG для Лабораторной работы №3.
Реализует динамический пайплайн с возможностью ручного выбора датасета
через Airflow UI (Params) для предотвращения перегрузки системы.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
import mlflow

from lab3.scripts.data_manager import DynamicDataManager
from lab3.scripts.task_detector import TaskDetector
from lab3.scripts.automl_trainer import DynamicAutoMLTrainer

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

# Чтение доступных датасетов для формирования выпадающего списка в UI
config_path = "/opt/airflow/project/lab3/config/dataset_configs.json"
try:
    with open(config_path, "r") as f:
        available_datasets = list(json.load(f).keys())
except Exception:
    available_datasets = ["bank_churn", "wine_quality"]

default_args = {
    'owner': 'student',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def task_load_and_preprocess(**context) -> dict:
    """Шаг 1: Динамическая загрузка и предобработка выбранного датасета."""
    # Получение датасета, выбранного пользователем при запуске DAG'а
    dataset_name = context['params']['dataset_name']
    logging.info(f"Инициализация обработки датасета: {dataset_name}")
    
    manager = DynamicDataManager()
    raw_df = manager.load_or_download_data(dataset_name)
    train_path, test_path, target_col = manager.preprocess_dataset(raw_df, dataset_name)
    
    configured_problem_type = manager.configs.get(dataset_name, {}).get("problem_type")
    
    return {
        'dataset_name': dataset_name,
        'train_path': train_path,
        'test_path': test_path,
        'target_col': target_col,
        'configured_problem_type': configured_problem_type
    }

def task_detect_task_type(**context) -> dict:
    """Шаг 2: Автоматическое определение типа задачи ML."""
    prep_data = context['ti'].xcom_pull(task_ids='load_and_preprocess_data')
    
    import pandas as pd
    train_df = pd.read_csv(prep_data['train_path'])
    target_series = train_df[prep_data['target_col']]
    
    task_type = TaskDetector.detect_task_type(
        target_series=target_series,
        configured_type=prep_data.get('configured_problem_type')
    )
    
    logging.info(f"Определен тип задачи для {prep_data['dataset_name']}: {task_type}")
    return {'task_type': task_type}

def task_run_automl(**context) -> dict:
    """Шаг 3: Запуск AutoML алгоритма для подбора гиперпараметров."""
    prep_data = context['ti'].xcom_pull(task_ids='load_and_preprocess_data')
    task_info = context['ti'].xcom_pull(task_ids='detect_task_type')
    
    trainer = DynamicAutoMLTrainer(
        dataset_name=prep_data['dataset_name'],
        task_type=task_info['task_type'],
        target_col=prep_data['target_col']
    )
    
    automl_results = trainer.train_and_optimize(
        train_path=prep_data['train_path'],
        test_path=prep_data['test_path']
    )
    
    logging.info(f"AutoML завершен. Победитель: {automl_results['winner_model']}")
    return automl_results

def task_log_summary(**context) -> None:
    """Шаг 4: Формирование итогового отчета AutoML в MLflow."""
    prep_data = context['ti'].xcom_pull(task_ids='load_and_preprocess_data')
    task_info = context['ti'].xcom_pull(task_ids='detect_task_type')
    automl_res = context['ti'].xcom_pull(task_ids='run_automl')
    
    report = {
        "dataset": prep_data['dataset_name'],
        "detected_task_type": task_info['task_type'],
        "champion_model": automl_res['winner_model'],
        "champion_metrics": automl_res['winner_metrics'],
        "all_candidates": automl_res['all_results']
    }
    
    report_file = f"/opt/airflow/data/{prep_data['dataset_name']}_automl_summary.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)
        
    mlflow.set_experiment(f"lab3_automl_{prep_data['dataset_name']}")
    with mlflow.start_run(run_name="00_automl_champion_summary", nested=True):
        mlflow.log_artifact(report_file)
        mlflow.log_params(automl_res['all_results'][automl_res['winner_model']]['best_params'])
        mlflow.log_metrics(automl_res['winner_metrics'])
        mlflow.set_tag("status", "completed")

with DAG(
    dag_id='dynamic_ml_pipeline',
    default_args=default_args,
    description='Lab 3: Dynamic Pipeline with UI Manual Selection',
    schedule=None,
    catchup=False,
    tags=['mlops', 'lab3', 'automl'],
    params={
        "dataset_name": Param(
            default="bank_churn",
            type="string",
            enum=available_datasets,
            description="Выберите датасет для запуска цикла AutoML"
        )
    }
) as dag:

    t1 = PythonOperator(task_id='load_and_preprocess_data', python_callable=task_load_and_preprocess)
    t2 = PythonOperator(task_id='detect_task_type', python_callable=task_detect_task_type)
    t3 = PythonOperator(task_id='run_automl', python_callable=task_run_automl)
    t4 = PythonOperator(task_id='log_automl_summary', python_callable=task_log_summary)

    t1 >> t2 >> t3 >> t4