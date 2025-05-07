import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
'''
Deeper explanations can be found in the Perceptron notebook.

A simple implementation of the Perceptron algorithm for binary classification.

    The Perceptron is a linear classifier that learns a decision boundary
    by iteratively updating weights based on misclassified examples.
    This implementation assumes target labels are -1 and 1 for internal
    processing during training, and predicts 0 or 1.

    Parameters
    ----------
    learning_rate : float, optional
        The step size for weight updates during training. Defaults to 0.01.
        A smaller learning rate might lead to slower convergence but can
        be more stable.
    n_iter : int, optional
        The number of passes over the training dataset (epochs).
        Defaults to 1000.

    Attributes
    ----------
    lr : float
        Learning rate for weight updates.
    n_iter : int
        Number of iterations (epochs) for training.
    weights : numpy.ndarray
        The weights learned during training. Shape is (n_features,).
        Initialized to zeros and updated during the `fit` method.
    bias : float
        The bias term learned during training. Initialized to zero and
        updated during the `fit` method.

    Methods
    -------
    fit(X, y)
        Trains the Perceptron model on the given training data.
    predict(X)
        Predicts class labels for new data instances.
    plot_decision_boundary(X, y)
        Plots the decision boundary after training.
'''

class Perceptron:
    
    def __init__(self, learning_rate=0.01, n_iter=1000):
        """
        Initializes the Perceptron classifier.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate (between 0.0 and 1.0) for weight updates.
            Controls the step size of the updates. Defaults to 0.01.
        n_iter : int, optional
            The number of passes over the training dataset (epochs).
            Defaults to 1000.
        """
        self.lr = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fits the Perceptron model to the training data.

        The weights and bias are updated iteratively based on the Perceptron
        learning rule for each misclassified sample.

        Parameters
        ----------
        X : numpy.ndarray
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features. Shape: (n_samples, n_features).
        y : numpy.ndarray
            Target values (class labels). Expected to be convertible to -1 or 1.
            Values <= 0 are treated as -1, and values > 0 are treated as 1
            for the internal update rule. Shape: (n_samples,).

        Returns
        -------
        self : Perceptron
            The fitted Perceptron model.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bias)
                if condition <= 0:
                    self.weights += self.lr * y_[idx] * x_i
                    self.bias += self.lr * y_[idx]

    def predict(self, X):
        '''Predicts class labels for input data X.

        The prediction is based on the sign of the linear combination of
        the input features, weights, and bias.

        Parameters
        ----------
        X : numpy.ndarray
            Input data for which to make predictions.
            Shape: (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Predicted class labels (0 or 1). Shape: (n_samples,).
            Returns 1 if the linear output is > 0, otherwise 0.
        '''
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output > 0, 1, 0)
    
    def plot_decision_boundary(self, X, y):
        """
        Plots the decision boundary learned by the Perceptron using mlxtend.

        Parameters
        ----------
        X : numpy.ndarray
            The input features, shape (n_samples, 2).
        y : numpy.ndarray
            The class labels, shape (n_samples,).
        """
        if X.shape[1] != 2:
            raise ValueError("plot_decision_boundary only works for 2D data.")
        
        class _WrapperModel:
            def __init__(self, perceptron):
                self.perceptron = perceptron
            def predict(self, X):
                return self.perceptron.predict(X)
        
        wrapper = _WrapperModel(self)
        
        plt.figure(figsize=(8, 6))
        plot_decision_regions(X, y, clf=wrapper, legend=2)
        plt.title('Perceptron Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
