"""Visualization utilities for polynomial regression."""

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt



def plot_regression_curve(
    X: np.ndarray,
    y: np.ndarray,
    X_plot: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Polynomial Regression Fit",
) -> Path:
    """
    Plot training data and predicted polynomial curve.

    Parameters
    ----------
    X, y:
        Training data (1D feature expected for visualization).
    X_plot:
        Sorted feature values used to draw the smooth curve.
    y_pred:
        Model predictions for ``X_plot``.
    output_path:
        Destination for the saved SVG file. Defaults to ``examples/polynomial_fit.svg``.
    title:
        Plot title.
    """

    if output_path is None:
        output_path = Path("examples/polynomial_fit.svg")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color="steelblue", alpha=0.7, label="Data")
    plt.plot(X_plot, y_pred, color="darkorange", linewidth=2.5, label="Model prediction")
    plt.xlabel("Feature X")
    plt.ylabel("Target y")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path
