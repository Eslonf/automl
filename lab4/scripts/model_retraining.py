"""
Модуль автоматического переобучения модели.
"""

import logging
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRetrainer:
    """Класс для дообучения модели и обновления ее версии в Registry."""
    
    def __init__(self, model_name: str = "bank_churn_model"):
        self.model_name = model_name
        self.client = MlflowClient()

    def retrain_model(self, train_path: str = "/opt/airflow/data/churn_processed/train.csv", 
                      test_path: str = "/opt/airflow/data/churn_processed/test.csv") -> tuple:
        """Переобучает модель и устанавливает алиас @champion новой версии."""
        logger.info("Старт процесса переобучения...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train, y_train = train_df.drop(columns=['Exited']), train_df['Exited']
        X_test, y_test = test_df.drop(columns=['Exited']), test_df['Exited']

        new_model = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            random_state=42,
            eval_metric="logloss"
        )
        new_model.fit(X_train, y_train)

        y_pred = new_model.predict(X_test)
        y_proba = new_model.predict_proba(X_test)[:, 1]

        new_f1 = float(f1_score(y_test, y_pred))
        new_acc = float(accuracy_score(y_test, y_pred))
        new_auc = float(roc_auc_score(y_test, y_proba))

        mlflow.set_experiment("lab4_monitoring_and_drift")
        with mlflow.start_run(run_name="retrained_production_model", nested=True) as run:
            mlflow.log_params(new_model.get_params())
            mlflow.log_metrics({"f1_score": new_f1, "accuracy": new_acc, "roc_auc": new_auc})
            mlflow.xgboost.log_model(new_model, artifact_path="model")
            run_id = run.info.run_id

        model_version = mlflow.register_model(f"runs:/{run_id}/model", self.model_name)
        self.client.set_registered_model_alias(self.model_name, "champion", model_version.version)
        
        logger.info(f"Новая версия v{model_version.version} успешно стала @champion!")
        return model_version.version, new_f1