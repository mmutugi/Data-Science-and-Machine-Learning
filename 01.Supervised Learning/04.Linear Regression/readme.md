# Linear Regression Classifier

An exploration of linear classification algorithms using Python and a synthetic data set of Matatus in Kenya. Designed for understanding the fundamentals of linear Regression and gradient methods.

## Features

-   Classification using algorithms that create linear decision boundaries.
-   Typically involves learning weights for features and a bias term.
-   Parameters often configurable (e.g., learning rate, regularization strength, depending on the specific algorithm).
-   Decision boundary visualization (a line or hyperplane) for 2D/3D data.
-   Model performance evaluation (accuracy, precision, recall, confusion matrix, etc.).

## What is a Linear Classifier?

A linear classifier is a type of classification algorithm that makes predictions based on a **linear predictor function** combining a set of weights with the feature vector. In simple terms, it separates data points belonging to different classes using a line (in 2 dimensions), a plane (in 3 dimensions), or a **hyperplane** (in higher dimensions).

The decision rule is typically based on the sign of $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$, where $\mathbf{x}$ is the input feature vector, $\mathbf{w}$ is the weight vector, and $b$ is the bias term. Different linear classifier algorithms (like Perceptron, Logistic Regression, Linear Regression) vary in how they determine the optimal weights ($\mathbf{w}$) and bias ($b$), often by optimizing different objective or loss functions.

## Getting Started

### Need

Make sure you have the following libraries installed, as they are commonly used for implementing or analyzing linear classifiers:

```bash
numpy         (For numerical operations)
matplotlib    (For plotting)
pandas        (For data handling)
mlxtend.plotting (Optional, helpful for decision boundary plots)
```

### File Structure

```bash
regression.ipynb   # main notebook with the algorithm.
readme.md  # THis file
matatu_data.csv # Synthetic Data set about Matatus in Kenya
.png #Image files used in the notebook
```
Authored by `Mark Munyi`