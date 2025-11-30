"""Run the full polynomial regression workflow."""

from pathlib import Path
import numpy as np

from app.data import generate_synthetic_data
from app.preprocess import split_data
from app.model import build_polynomial_regression, predict, train_model
from app.evaluate import regression_metrics
from app.visualize import plot_regression_curve


def run_pipeline(degree: int = 3) -> dict:
    """
    Execute the end-to-end workflow and return evaluation metrics.
    """

    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_polynomial_regression(degree=degree)
    model = train_model(model, X_train, y_train)

    y_pred = predict(model, X_test)
    metrics = regression_metrics(y_test, y_pred)

    # Prepare smooth curve for visualization (only works for 1D input)
    X_plot = np.linspace(X.min() - 5, X.max() + 5, 300).reshape(-1, 1)
    y_plot = predict(model, X_plot)
    plot_regression_curve(X, y, X_plot, y_plot, output_path=Path("examples/polynomial_fit.svg"))

    return metrics


def main() -> None:
    metrics = run_pipeline(degree=3)
    print("Polynomial Regression Metrics:")
    for name, value in metrics.items():
        print(f"- {name}: {value:.3f}")


if __name__ == "__main__":
    main()
