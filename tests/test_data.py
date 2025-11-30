import numpy as np
from app.data import generate_synthetic_data


def test_generate_synthetic_data_shapes():
    X, y = generate_synthetic_data(n_samples=50, random_state=0)
    assert X.shape == (50, 1)
    assert y.shape == (50,)
    # Ensure nonlinearity added
    assert np.var(y) > 0
