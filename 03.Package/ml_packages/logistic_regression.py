import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

class LogisticRegression:
    """
    Logistic Regression classifier with stochastic gradient descent and support for cross-entropy.

    Attributes
    ----------
    errors_ : list
        List containing the cost (MSE or cross-entropy) computed after each epoch.

    Methods
    -------
    train(X, y, alpha=0.005, epochs=50, loss='mse')
        Trains the model using stochastic gradient descent.
        Supported loss: 'mse', 'cross_entropy'

    predict_proba(X)
        Predicts probabilities (sigmoid output) for input X.

    predict(X)
        Predicts hard labels (0 or 1) for input X.

    plot_cost_function()
        Plots the cost function over training epochs.

    plot_decision_boundary(X, y, xstring='x', ystring='y')
        Visualizes the decision boundary on a 2D feature space.
    """
    def __init__(self):
        self.w_ = None
        self.errors_ = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _cross_entropy(self, prediction, target):
        epsilon = 1e-15 
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        return - (target * np.log(prediction) + (1 - target) * np.log(1 - prediction))

    def train(self, X, y, alpha=0.005, epochs=100, loss='mse'):
        self.w_ = np.random.rand(1 + X.shape[1])
        N = X.shape[0]
        self.errors_ = []

        for _ in range(epochs):
            errors = 0
            for xi, target in zip(X, y):
                pred = self._sigmoid(np.dot(xi, self.w_[:-1]) + self.w_[-1])
                error = pred - target
                self.w_[:-1] -= alpha * error * xi
                self.w_[-1] -= alpha * error
                if loss == 'mse':
                    errors += 0.5 * error ** 2
                elif loss == 'cross_entropy':
                    errors += self._cross_entropy(pred, target)
            self.errors_.append(errors / N)
        return self

    def predict_proba(self, X):
        return self._sigmoid(np.dot(X, self.w_[:-1]) + self.w_[-1])

    def predict(self, X):
        probs = self.predict_proba(X)
        if X.ndim == 1:
            return int(probs > 0.5)
        return np.where(probs > 0.5, 1, 0)

    def plot_cost_function(self):
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(self.errors_) + 1), self.errors_, label="Cost function")
        plt.xlabel("Epochs", fontsize=15)
        plt.ylabel("Cost", fontsize=15)
        plt.legend(fontsize=15)
        plt.title("Cost Calculated After Each Epoch During Training", fontsize=18)
        plt.show()

    def plot_decision_boundary(self, X, y, xstring="x", ystring="y"):
        plt.figure(figsize=(10, 8))
        plot_decision_regions(X, y, clf=self)
        plt.title("Logistic Regression Decision Boundary", fontsize=18)
        plt.xlabel(xstring, fontsize=15)
        plt.ylabel(ystring, fontsize=15)
        plt.show()


    