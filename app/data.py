"""Data generation utilities for polynomial regression examples."""

from typing import Tuple
import numpy as np
from sklearn.datasets import make_regression


def generate_synthetic_data(
    n_samples: int = 200,
    noise: float = 8.0,
    random_state: int | None = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a simple nonlinear regression dataset using ``make_regression``.

    The base dataset is linear; we apply a quadratic transformation to
    introduce curvature so polynomial regression has an advantage.

    Parameters
    ----------
    n_samples:
        Number of data points to generate.
    noise:
        Standard deviation of the Gaussian noise added to the targets.
    random_state:
        Reproducibility seed.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Feature matrix ``X`` with shape ``(n_samples, 1)`` and target vector ``y``.
    """

    X, y = make_regression(
        n_samples=n_samples,
        n_features=1,
        n_informative=1,
        noise=noise,
        random_state=random_state,
        bias=30.0,
    )

    # Introduce a nonlinear relationship: y = y_linear + 0.8 * x^2
    y_nonlinear = y + 0.8 * np.square(X[:, 0])
    return X, y_nonlinear
