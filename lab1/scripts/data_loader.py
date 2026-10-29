"""
Модуль для загрузки данных из внешних источников (Kaggle API).
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
    """
    Аутентифицируется в Kaggle, скачивает и распаковывает целевой датасет.
    
    Returns:
        tuple: (Путь к сырому CSV файлу, количество строк, количество колонок)
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    logger.info("Инициализация Kaggle API и аутентификация...")
    api = KaggleApi()
    api.authenticate()
    
    logger.info(f"Скачивание датасета {DATASET_NAME}...")
    api.dataset_download_files(DATASET_NAME, path=DOWNLOAD_DIR, unzip=True)
    
    csv_file = None
    for file in os.listdir(DOWNLOAD_DIR):
        if file.endswith(".csv"):
            csv_file = os.path.join(DOWNLOAD_DIR, file)
            break
            
    if not csv_file:
        raise FileNotFoundError("CSV файл не обнаружен в скачанном архиве.")
        
    df = pd.read_csv(csv_file)
    logger.info(f"Успешная загрузка. Размерность сырых данных: {df.shape}")
    
    raw_path = "/opt/airflow/data/churn_raw.csv"
    df.to_csv(raw_path, index=False)
    
    return raw_path, df.shape[0], df.shape[1]