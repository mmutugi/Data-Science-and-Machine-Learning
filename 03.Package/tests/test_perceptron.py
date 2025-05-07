import numpy as np
import pytest
from ml_packages.perceptron import Perceptron

@pytest.fixture
def sample_data():

    """
    Pytest fixture to provide sample data for Perceptron testing.

    This fixture generates a simple dataset (XOR-like pattern)
    intended for binary classification tasks.

    Returns
    -------
    tuple
        A tuple containing:
            - X (numpy.ndarray): Feature matrix of shape (4, 2).
            - y (numpy.ndarray): Target labels of shape (4,).
                                 Labels are 0 or 1.
    """
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 1, 0])
    return X, y

def test_fit_predict(sample_data):
    """
    Tests the basic `fit` and `predict` functionality of the Perceptron classifier.

    This test verifies that:
    1. The `Perceptron.fit()` method runs without raising an error.
    2. The `Perceptron.predict()` method runs without raising an error after fitting.
    3. The shape of the predictions array matches the shape of the input target labels.
    4. All predicted labels are binary (i.e., belong to the set {0, 1}).

    Note: This test uses a small number of iterations (`n_iter=10`) and an
    XOR-like dataset which is not linearly separable. Therefore, it primarily
    checks for operational correctness (runs without crashing, correct output
    format) rather than high classification accuracy.

    Parameters
    ----------
    sample_data : tuple
        A tuple (X, y) provided by the `sample_data` Pytest fixture, where
        X is the feature matrix and y is the array of target labels.
    """
    X, y = sample_data
    clf = Perceptron(learning_rate=0.1, n_iter=10)
    clf.fit(X, y)
    predictions = clf.predict(X)
    assert predictions.shape == y.shape
    assert np.all(np.isin(predictions, [0, 1]))
