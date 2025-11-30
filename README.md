# Polynomial Regression Tutorial and Template

A beginner-friendly walkthrough of polynomial regression using scikit-learn. The project shows how to generate synthetic nonlinear data, expand features with `PolynomialFeatures`, train a `LinearRegression` model, evaluate performance, and visualize the resulting curve.

## Why Polynomial Regression?
Linear regression fits straight lines. Polynomial regression keeps the model linear in its parameters but expands input features with powers (e.g., x, x², x³) to capture curvature. Increasing the degree adds flexibility but can overfit—higher degrees memorize noise while lower degrees may underfit.

## Project Flow
1. **Data loading** – generate a synthetic nonlinear dataset with scikit-learn.
2. **Preprocessing** – split into train/test (scaling is optional for 1D data).
3. **Feature engineering** – create polynomial terms up to the chosen degree.
4. **Model training** – fit a linear regression model on the expanded features.
5. **Evaluation** – report MAE, MSE, RMSE, and R².
6. **Visualization** – plot data points and the predicted polynomial curve.

## Dataset
The code generates a 1D synthetic regression dataset using `make_regression`, then adds a quadratic term to introduce nonlinearity. You can tweak the sample size, noise, and random seed inside `app/data.py`.

## Repository Structure
```
app/
  data.py          # synthetic data generation
  preprocess.py    # train/test split (scaling guidance included)
  model.py         # PolynomialFeatures + LinearRegression pipeline
  evaluate.py      # regression metrics helpers
  visualize.py     # SVG plot of data and fitted curve
  main.py          # end-to-end script
notebooks/
  demo_polynomial_regression.ipynb  # hands-on tutorial
examples/
  README_examples.md                # saved plots live here
  polynomial_fit.svg                # generated after running main.py
tests/
  test_data.py     # dataset sanity checks
  test_model.py    # pipeline fit/predict
  test_evaluate.py # metric calculations
requirements.txt
Dockerfile
LICENSE
```

## Quickstart
```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python app/main.py

# Execute tests
pytest

# Open the notebook
jupyter notebook notebooks/demo_polynomial_regression.ipynb
```

## Docker
```bash
docker build -t polynomial-regression .
docker run --rm polynomial-regression
```

## Future Extensions
- Search for the best polynomial degree with cross-validation
- Add regularization (Ridge/Lasso) to reduce overfitting
- Support multi-dimensional inputs with scaling
- Log metrics with MLflow or Weights & Biases

## License
MIT License. See [LICENSE](LICENSE).
