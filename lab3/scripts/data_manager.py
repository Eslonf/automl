"""
Модуль динамического управления данными (загрузка и универсальная очистка).
"""

import os
import json
import logging
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicDataManager:
    """Класс для работы с конфигурациями датасетов и их обработкой."""
    
    def __init__(self, config_path: str = "/opt/airflow/project/lab3/config/dataset_configs.json"):
        with open(config_path, "r") as f:
            self.configs = json.load(f)

    def load_or_download_data(self, dataset_name: str) -> pd.DataFrame:
        """Загружает датасет с Kaggle на основе конфигурации."""
        if dataset_name not in self.configs:
            raise ValueError(f"Датасет '{dataset_name}' отсутствует в конфигурации.")

        cfg = self.configs[dataset_name]
        download_dir = f"/opt/airflow/data/{dataset_name}_raw"
        os.makedirs(download_dir, exist_ok=True)

        logger.info(f"Загрузка датасета {dataset_name} ({cfg['kaggle_dataset']})...")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(cfg["kaggle_dataset"], path=download_dir, unzip=True)

        csv_file = next((os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.endswith(".csv")), None)
        if not csv_file:
            raise FileNotFoundError(f"CSV файл не найден в {download_dir}")

        df = pd.read_csv(csv_file)
        logger.info(f"Датасет загружен. Размерность: {df.shape}")
        return df

    def preprocess_dataset(self, df: pd.DataFrame, dataset_name: str) -> tuple:
        """Универсальная предобработка данных (очистка, кодирование, масштабирование)."""
        target_col = self.configs[dataset_name]["target_column"]
        df_clean = df.copy()

        # Удаление ID-колонок
        for col in df_clean.columns:
            if col != target_col and df_clean[col].nunique() == len(df_clean):
                df_clean.drop(columns=[col], inplace=True)

        for col in ['Surname', 'Name', 'Id', 'Id_']:
            if col in df_clean.columns and col != target_col:
                df_clean.drop(columns=[col], inplace=True)

        # Импутация
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                fill_val = df_clean[col].mode()[0] if df_clean[col].dtype in ['object', 'category'] else df_clean[col].median()
                df_clean[col].fillna(fill_val, inplace=True)

        # Кодирование
        for col in df_clean.columns:
            if col != target_col and df_clean[col].dtype in ['object', 'category']:
                df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))

        X, y = df_clean.drop(columns=[target_col]), df_clean[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Масштабирование
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        save_dir = f"/opt/airflow/data/dynamic_{dataset_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
        train_df[target_col] = y_train.values
        test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
        test_df[target_col] = y_test.values

        train_path, test_path = f"{save_dir}/train.csv", f"{save_dir}/test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Предобработка завершена: Train {train_df.shape}, Test {test_df.shape}")
        return train_path, test_path, target_col