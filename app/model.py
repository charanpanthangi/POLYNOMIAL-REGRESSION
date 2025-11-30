"""Model definition for polynomial regression."""

from typing import Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures



def build_polynomial_regression(degree: int = 3) -> Pipeline:
    """
    Create a scikit-learn pipeline with polynomial expansion and linear regression.

    Parameters
    ----------
    degree:
        Degree of the polynomial features. Higher values allow more flexible curves
        but may overfit, especially with limited data.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline that maps the input feature into polynomial terms then fits
        a linear regression model on the expanded space.
    """

    pipeline = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("model", LinearRegression()),
        ]
    )
    return pipeline


def train_model(
    model: Pipeline, X_train: np.ndarray, y_train: np.ndarray
) -> Pipeline:
    """Fit the pipeline on the training data and return it."""

    model.fit(X_train, y_train)
    return model


def predict(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Run predictions using the fitted model."""

    return model.predict(X)
