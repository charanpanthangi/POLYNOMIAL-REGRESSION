import numpy as np
from app.model import build_polynomial_regression, train_model, predict
from app.data import generate_synthetic_data
from app.preprocess import split_data


def test_model_fit_and_predict():
    X, y = generate_synthetic_data(n_samples=80, random_state=1)
    X_train, X_test, y_train, _ = split_data(X, y, test_size=0.25, random_state=1)
    model = build_polynomial_regression(degree=3)
    trained = train_model(model, X_train, y_train)
    preds = predict(trained, X_test)
    assert preds.shape == (X_test.shape[0],)
    # Predictions should not all be identical
    assert np.std(preds) > 0
