import numpy as np
import pytest
from app.evaluate import regression_metrics


def test_regression_metrics_output():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    metrics = regression_metrics(y_true, y_pred)

    assert set(metrics.keys()) == {
        "mean_absolute_error",
        "mean_squared_error",
        "root_mean_squared_error",
        "r2_score",
    }
    assert metrics["mean_absolute_error"] >= 0
    assert metrics["root_mean_squared_error"] == pytest.approx(np.sqrt(metrics["mean_squared_error"]))
