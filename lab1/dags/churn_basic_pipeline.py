"""
DAG для Лабораторной работы №1.
Реализует базовый пайплайн загрузки и предобработки данных с логированием в MLflow.
"""

import os
from datetime import datetime, timedelta
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow

from lab1.scripts.data_loader import download_churn_data
from lab1.scripts.data_preprocessor import preprocess_churn_data

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("lab1_bank_customer_churn")

default_args = {
    'owner': 'student',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def task_download() -> str:
    """
    Задача Airflow: Загрузка сырых данных.
    Логирует параметры источника и базовые метрики размерности датасета.
    """
    with mlflow.start_run(run_name="01_data_download", nested=True):
        raw_path, rows, cols = download_churn_data()
        
        mlflow.log_param("dataset_name", "Bank Customer Churn")
        mlflow.log_param("source", "Kaggle")
        mlflow.log_metric("raw_rows_count", rows)
        mlflow.log_metric("raw_columns_count", cols)
        
        return raw_path

def task_preprocess() -> None:
    """
    Задача Airflow: Предобработка данных.
    Логирует стратегии очистки и итоговую размерность выборок.
    """
    with mlflow.start_run(run_name="02_data_preprocessing", nested=True):
        train_path, test_path, train_shape, test_shape = preprocess_churn_data()
        
        mlflow.log_param("strategy_missing", "median_mode")
        mlflow.log_param("strategy_encoding", "label_encoder")
        mlflow.log_param("strategy_scaling", "standard_scaler")
        mlflow.log_metric("train_samples", train_shape[0])
        mlflow.log_metric("test_samples", test_shape[0])
        mlflow.log_metric("features_count", train_shape[1] - 1)

def task_summary() -> None:
    """
    Задача Airflow: Формирование отчета.
    Рассчитывает дисбаланс классов и сохраняет текстовый артефакт в MLflow.
    """
    with mlflow.start_run(run_name="03_data_summary", nested=True):
        train_df = pd.read_csv("/opt/airflow/data/churn_processed/train.csv")
        sample_path = "/opt/airflow/data/processed_data_sample.txt"
        
        # Генерация текстового отчета
        with open(sample_path, "w") as f:
            f.write("=== PROCESSED DATA SAMPLE ===\n")
            f.write(train_df.head(10).to_string())
            f.write("\n\n=== INFO ===\n")
            train_df.info(buf=f)
            
        mlflow.log_artifact(sample_path)
        
        # Расчет и логирование доли положительного класса
        ratio = train_df['Exited'].mean()
        mlflow.log_metric("exited_class_ratio", float(ratio))

with DAG(
    dag_id='churn_basic_pipeline',
    default_args=default_args,
    description='Lab 1: Baseline pipeline for data preparation',
    schedule=None,
    catchup=False,
    tags=['mlops', 'lab1']
) as dag:

    t1 = PythonOperator(task_id='download_data', python_callable=task_download)
    t2 = PythonOperator(task_id='preprocess_data', python_callable=task_preprocess)
    t3 = PythonOperator(task_id='log_summary', python_callable=task_summary)

    t1 >> t2 >> t3