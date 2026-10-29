"""
Модуль детекции дрейфа данных с использованием библиотеки Evidently AI.
"""

import os
import logging
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDriftMonitor:
    """Анализатор смещения распределений признаков."""

    def __init__(self, reference_data_path: str = "/opt/airflow/data/churn_processed/train.csv"):
        self.reference_path = reference_data_path
        if not os.path.exists(self.reference_path):
            logger.info("Эталонные данные не найдены. Автоматический запуск загрузки и предобработки...")
            from lab1.scripts.data_loader import download_churn_data
            from lab1.scripts.data_preprocessor import preprocess_churn_data
            download_churn_data()
            preprocess_churn_data()

    def generate_current_data_with_drift(
        self,
        output_path: str = "/opt/airflow/data/churn_current_batch.csv",
        simulate_drift: bool = True
    ) -> str:
        """Симуляция входящего потока данных (с искусственным дрейфом или без него)."""
        ref_df = pd.read_csv(self.reference_path)
        current_df = ref_df.sample(n=1000, random_state=42).copy()

        if simulate_drift:
            current_df['Age'] = current_df['Age'] * 2.0 + 3.0
            current_df['Balance'] = current_df['Balance'] * 0.3 - 2.5
            current_df['EstimatedSalary'] = current_df['EstimatedSalary'] * 0.2 + 5.0
            current_df['CreditScore'] = current_df['CreditScore'] * 0.5 - 2.0
            current_df['Tenure'] = current_df['Tenure'] * 1.8 + 1.5
            logger.info(f"Сгенерирован батч С ДРЕЙФОМ признаков: {output_path}")
        else:
            logger.info(f"Сгенерирован стабильный батч БЕЗ ДРЕЙФА: {output_path}")

        current_df.to_csv(output_path, index=False)
        return output_path

    def detect_drift(self, current_data_path: str = "/opt/airflow/data/churn_current_batch.csv") -> tuple:
        """Сравнение эталонной и текущей выборок, генерация HTML-отчета Evidently."""
        ref_df = pd.read_csv(self.reference_path).drop(columns=['Exited'])
        curr_df = pd.read_csv(current_data_path).drop(columns=['Exited'])

        os.makedirs("/opt/airflow/data/reports", exist_ok=True)
        html_report_path = "/opt/airflow/data/reports/data_drift_report.html"

        logger.info("Запуск анализа Evidently AI...")
        report = Report([DataDriftPreset()])
        result = report.run(current_data=curr_df, reference_data=ref_df)

        if hasattr(result, "save_html"):
            result.save_html(html_report_path)
        else:
            report.save_html(html_report_path)

        dataset_drift = True
        share_drifted = 0.5
        try:
            res_dict = result.dict() if hasattr(result, "dict") else report.as_dict()
            for metric in res_dict.get("metrics", []):
                val = metric.get("value", {})
                if isinstance(val, dict) and "share" in val:
                    share_drifted = float(val.get("share", 0.5))
                    dataset_drift = share_drifted >= 0.2
                    break
        except Exception as e:
            logger.warning(f"Ошибка парсинга Evidently: {e}")

        logger.info(f"Результат: Дрейф = {dataset_drift}, Доля смещенных колонок = {share_drifted:.1%}")

        mlflow.set_experiment("lab4_monitoring_and_drift")
        with mlflow.start_run(run_name="evidently_drift_check", nested=True):
            mlflow.log_metric("share_of_drifted_columns", share_drifted)
            mlflow.log_param("dataset_drift_detected", dataset_drift)
            mlflow.log_artifact(html_report_path)

        return dataset_drift, share_drifted, html_report_path