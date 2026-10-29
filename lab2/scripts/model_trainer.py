"""
Модуль обучения, оценки и регистрации моделей машинного обучения.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Класс для управления жизненным циклом обучения моделей."""
    
    def __init__(self, experiment_name: str = "lab2_model_comparison"):
        self.experiment_name = experiment_name
        self.models = {
            "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            "xgboost": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric="logloss")
        }

    def load_data(self, train_path: str = "/opt/airflow/data/churn_processed/train.csv", 
                  test_path: str = "/opt/airflow/data/churn_processed/test.csv") -> tuple:
        """Загрузка подготовленных данных с диска."""
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        return (
            train_df.drop(columns=['Exited']), train_df['Exited'],
            test_df.drop(columns=['Exited']), test_df['Exited']
        )

    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> str:
        """Генерация и сохранение графика матрицы ошибок."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        os.makedirs("/opt/airflow/data/artifacts", exist_ok=True)
        plot_path = f"/opt/airflow/data/artifacts/confusion_matrix_{model_name}.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def plot_feature_importance(self, model, feature_names: list, model_name: str) -> str:
        """Генерация и сохранение графика важности признаков."""
        if not hasattr(model, 'feature_importances_'):
            return None
            
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(8, 6))
        plt.title(f'Feature Importances - {model_name}')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        
        os.makedirs("/opt/airflow/data/artifacts", exist_ok=True)
        plot_path = f"/opt/airflow/data/artifacts/feature_importance_{model_name}.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def train_and_evaluate(self) -> tuple:
        """
        Обучает заданные модели, вычисляет метрики, логирует результаты в MLflow
        и определяет лучшую модель на основе F1-score.
        """
        X_train, y_train, X_test, y_test = self.load_data()
        feature_names = list(X_train.columns)
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Старт обучения модели: {name}...")
            with mlflow.start_run(run_name=f"train_{name}", nested=True) as run:
                # Обучение и предсказание
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Расчет метрик
                metrics = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                    "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
                    "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else 0.0
                }
                
                # Логирование в MLflow
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.set_tag("model_type", name)
                
                # Артефакты
                mlflow.log_artifact(self.plot_confusion_matrix(y_test, y_pred, name))
                fi_path = self.plot_feature_importance(model, feature_names, name)
                if fi_path:
                    mlflow.log_artifact(fi_path)
                    
                # Сохранение весов
                if name == "xgboost":
                    mlflow.xgboost.log_model(model, artifact_path="model")
                else:
                    mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="cloudpickle")
                    
                results[name] = {
                    "metrics": metrics,
                    "run_id": run.info.run_id,
                    "model_uri": f"runs:/{run.info.run_id}/model"
                }
                logger.info(f"Модель {name} обучена. F1: {metrics['f1_score']:.4f}")
                
        # Определение победителя
        best_model_name = max(results, key=lambda k: results[k]['metrics']['f1_score'])
        best_run_id = results[best_model_name]['run_id']
        
        logger.info(f"🏆 Лучшая модель: {best_model_name} (F1: {results[best_model_name]['metrics']['f1_score']:.4f})")
        return results, best_model_name, best_run_id

    def register_best_model(self, model_name: str, run_id: str, registered_model_name: str = "bank_churn_model") -> str:
        """
        Регистрирует модель в MLflow Model Registry и устанавливает современный алиас @champion.
        """
        logger.info(f"Регистрация {model_name} в Model Registry под именем {registered_model_name}...")
        
        client = MlflowClient()
        model_uri = f"runs:/{run_id}/model"
        
        # Создание новой версии в реестре
        model_version = mlflow.register_model(model_uri, registered_model_name)
        
        client.set_registered_model_alias(registered_model_name, "champion", model_version.version)
        logger.info(f"Модели {registered_model_name} (v{model_version.version}) присвоен алиас @champion!")
                
        return model_version.version