"""Preprocessing helpers for the polynomial regression workflow."""

from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split


def split_data(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int | None = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split features and targets into train and test sets.

    For polynomial regression on a single feature, scaling is often optional because
    ``PolynomialFeatures`` simply expands powers of the input. However, if you add
    more features with very different ranges, consider applying ``StandardScaler``
    before fitting to help the optimizer.
    """

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
