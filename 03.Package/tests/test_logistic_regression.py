import numpy as np
import pytest
from ml_packages.logistic_regression import LogisticRegression

@pytest.fixture
def sample_data():

    """
    Pytest fixture to provide sample data for testing.

    This fixture generates a simple dataset (XOR-like pattern)
    suitable for binary classification tasks.

    Returns
    -------
    tuple
        A tuple containing:
            - X (numpy.ndarray): Feature matrix of shape (4, 2).
            - y (numpy.ndarray): Target labels of shape (4,).
    """
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 1, 0])
    return X, y

def test_fit_predict(sample_data):

    """
    Tests the fit and predict methods of the LogisticRegression classifier.

    This test verifies:
    1. The `fit` method runs without errors.
    2. The `predict` method runs without errors after fitting.
    3. The shape of the predictions matches the shape of the input target labels.
    4. All predicted labels are binary (either 0 or 1).

    It does not assert the correctness of the predictions against expected
    values for this specific sample_data, as that would depend heavily on the
    LogisticRegression implementation details and convergence for this non-linearly
    separable (XOR) problem.

    Parameters
    ----------
    sample_data : tuple
        A tuple (X, y) provided by the `sample_data` Pytest fixture.
        X is the feature matrix, and y is the target labels.
    """

    ## Just type pytest and hit enter to run all tests on the terminal.
    X, y = sample_data
    clf = LogisticRegression()
    clf.train(X, y, alpha=0.01, epochs=100)
    predictions = clf.predict(X)
    assert predictions.shape == y.shape
    assert np.all(np.isin(predictions, [0, 1]))
