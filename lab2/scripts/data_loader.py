"""
Модуль загрузки данных для Лабораторной работы №2.
"""

import os
import logging
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "barelydedicated/bank-customer-churn-modeling"
DOWNLOAD_DIR = "/opt/airflow/data/churn_data"

def download_churn_data() -> tuple:
    """Загрузка данных с Kaggle API во внутреннее хранилище."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET_NAME, path=DOWNLOAD_DIR, unzip=True)
    
    csv_file = next((os.path.join(DOWNLOAD_DIR, f) for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".csv")), None)
    if not csv_file:
        raise FileNotFoundError("CSV файл не найден в директории загрузки.")
        
    df = pd.read_csv(csv_file)
    raw_path = "/opt/airflow/data/churn_raw.csv"
    df.to_csv(raw_path, index=False)
    
    logger.info(f"Данные успешно загружены. Размер: {df.shape}")
    return raw_path, df.shape[0], df.shape[1]