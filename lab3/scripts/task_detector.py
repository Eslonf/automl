"""
Модуль эвристического определения типа задачи машинного обучения.
"""

import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskDetector:
    """Анализатор целевой переменной для определения типа задачи ML."""
    
    @staticmethod
    def detect_task_type(target_series: pd.Series, configured_type: str = None) -> str:
        """
        Определяет тип задачи: регрессия, бинарная или многоклассовая классификация.
        """
        if configured_type:
            logger.info(f"Используется тип задачи из конфигурации: '{configured_type}'")
            return configured_type

        clean_target = target_series.dropna()
        n_unique = clean_target.nunique()
        dtype = clean_target.dtype

        if np.issubdtype(dtype, np.floating) and n_unique > 10:
            task = "regression"
        elif n_unique == 2:
            task = "binary_classification"
        elif n_unique <= 20 or dtype == 'object':
            task = "multiclass_classification"
        else:
            task = "regression"

        logger.info(f"Автоматически определен тип задачи '{task}' (уникальных значений: {n_unique})")
        return task