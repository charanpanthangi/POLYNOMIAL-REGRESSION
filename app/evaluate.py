"""Model evaluation helpers."""

from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute common regression metrics.

    Returns a dictionary with:
    - mean_absolute_error (MAE)
    - mean_squared_error (MSE)
    - root_mean_squared_error (RMSE)
    - r2_score (R²)
    """

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {
        "mean_absolute_error": mean_absolute_error(y_true, y_pred),
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "r2_score": r2_score(y_true, y_pred),
    }
