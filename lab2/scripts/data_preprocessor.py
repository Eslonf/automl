"""
Модуль предобработки данных для Лабораторной работы №2.
"""

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_churn_data(raw_data_path: str = "/opt/airflow/data/churn_raw.csv") -> tuple:
    """Очистка, кодирование и масштабирование признаков датасета."""
    df = pd.read_csv(raw_data_path)
    
    # Удаление служебных колонок
    cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Импутация пропусков
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            fill_val = df[col].mode()[0] if df[col].dtype in ['object', 'category'] else df[col].median()
            df[col] = df[col].fillna(fill_val)
                
    # Кодирование категорий
    for col in ['Geography', 'Gender']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    
    # Сплитование
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Масштабирование
    scaler = StandardScaler()
    num_cols = [c for c in ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary'] if c in X_train.columns]
    
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # Сохранение
    processed_dir = "/opt/airflow/data/churn_processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    
    pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)
    
    logger.info(f"Предобработка завершена. Train: {X_train.shape}, Test: {X_test.shape}")
    return train_path, test_path