import numpy as np
import pytest
from ml_packages.perceptron import Perceptron

@pytest.fixture
def sample_data():
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 1, 0])
    return X, y

def test_fit_predict(sample_data):
    X, y = sample_data
    clf = Perceptron(learning_rate=0.1, n_iter=10)
    clf.fit(X, y)
    predictions = clf.predict(X)
    assert predictions.shape == y.shape
    assert np.all(np.isin(predictions, [0, 1]))
