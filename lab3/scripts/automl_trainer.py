"""
Модуль реализации AutoML (автоматического подбора моделей и гиперпараметров).
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicAutoMLTrainer:
    """Оркестратор подбора гиперпараметров в зависимости от типа задачи."""
    
    def __init__(self, dataset_name: str, task_type: str, target_col: str):
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.target_col = target_col

    def _get_candidate_pipelines(self) -> dict:
        """Возвращает словарь моделей и сеток гиперпараметров для поиска."""
        if self.task_type in ["binary_classification", "multiclass_classification"]:
            return {
                "logistic_regression": {
                    "model": LogisticRegression(max_iter=1000, random_state=42),
                    "params": {"C": [0.01, 0.1, 1.0, 10.0]}
                },
                "random_forest": {
                    "model": RandomForestClassifier(random_state=42),
                    "params": {"n_estimators": [50, 100], "max_depth": [5, 10, None]}
                },
                "xgboost": {
                    "model": XGBClassifier(random_state=42, eval_metric="logloss"),
                    "params": {"n_estimators": [50, 100], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]}
                }
            }
        else:
            return {
                "ridge_regression": {
                    "model": Ridge(random_state=42),
                    "params": {"alpha": [0.1, 1.0, 10.0, 100.0]}
                },
                "random_forest_regressor": {
                    "model": RandomForestRegressor(random_state=42),
                    "params": {"n_estimators": [50, 100], "max_depth": [5, 10]}
                },
                "xgboost_regressor": {
                    "model": XGBRegressor(random_state=42),
                    "params": {"n_estimators": [50, 100], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]}
                }
            }

    def train_and_optimize(self, train_path: str, test_path: str) -> dict:
        """Запускает RandomizedSearchCV для всех кандидатов и выбирает победителя."""
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train, y_train = train_df.drop(columns=[self.target_col]), train_df[self.target_col]
        X_test, y_test = test_df.drop(columns=[self.target_col]), test_df[self.target_col]

        if self.task_type in ["binary_classification", "multiclass_classification"]:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

        pipelines = self._get_candidate_pipelines()
        results = {}

        mlflow.set_experiment(f"lab3_automl_{self.dataset_name}")

        for name, p_data in pipelines.items():
            logger.info(f"AutoML: Оптимизация гиперпараметров для {name}...")
            
            search = RandomizedSearchCV(
                estimator=p_data["model"],
                param_distributions=p_data["params"],
                n_iter=3,
                cv=3,
                random_state=42,
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            y_pred = best_model.predict(X_test)

            metrics = {}
            if self.task_type == "binary_classification":
                y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
                metrics = {
                    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else 0.0
                }
                score_to_compare = metrics["f1"]
            elif self.task_type == "multiclass_classification":
                metrics = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0))
                }
                score_to_compare = metrics["accuracy"]
            else:
                metrics = {
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "r2": float(r2_score(y_test, y_pred))
                }
                score_to_compare = -metrics["rmse"]

            with mlflow.start_run(run_name=f"automl_{name}", nested=True) as run:
                mlflow.log_params(search.best_params_)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(best_model, artifact_path="model", serialization_format="cloudpickle")

                results[name] = {
                    "best_params": search.best_params_,
                    "metrics": metrics,
                    "score_to_compare": score_to_compare,
                    "run_id": run.info.run_id
                }

        best_name = max(results, key=lambda k: results[k]["score_to_compare"])
        champion_run_id = results[best_name]["run_id"]
        logger.info(f"🏆 AutoML Победитель: {best_name}")

        reg_model_name = f"{self.dataset_name}_automl_champion"
        client = MlflowClient()
        model_version = mlflow.register_model(f"runs:/{champion_run_id}/model", reg_model_name)
        client.set_registered_model_alias(reg_model_name, "champion", model_version.version)

        return {
            "winner_model": best_name,
            "winner_run_id": champion_run_id,
            "winner_metrics": results[best_name]["metrics"],
            "all_results": results
        }