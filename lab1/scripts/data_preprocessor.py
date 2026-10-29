"""
Модуль для очистки, трансформации и разделения данных.
"""

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_churn_data(raw_data_path: str = "/opt/airflow/data/churn_raw.csv") -> tuple:
    """
    Выполняет полный цикл предобработки сырых данных:
    очистка, импутация, кодирование, сплитование и масштабирование.
    
    Args:
        raw_data_path (str): Путь к исходному CSV файлу.
        
    Returns:
        tuple: (Путь к train, Путь к test, Форма train, Форма test)
    """
    df = pd.read_csv(raw_data_path)
    logger.info(f"Старт предобработки. Исходная размерность: {df.shape}")
    
    # Шаг 1: Удаление неинформативных признаков
    cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Шаг 2: Обработка пропущенных значений
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['object', 'category']:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
                
    # Шаг 3: Кодирование категориальных признаков
    cat_cols = ['Geography', 'Gender']
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            
    # Шаг 4: Разделение на признаки (X) и целевую переменную (y)
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    
    # Стратифицированное разбиение выборки (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Шаг 5: Z-масштабирование числовых признаков
    scaler = StandardScaler()
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    num_cols = [c for c in num_cols if c in X_train.columns]
    
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # Шаг 6: Сохранение обработанных данных
    processed_dir = "/opt/airflow/data/churn_processed"
    os.makedirs(processed_dir, exist_ok=True)
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Предобработка завершена. Train: {train_df.shape}, Test: {test_df.shape}")
    return train_path, test_path, train_df.shape, test_df.shape