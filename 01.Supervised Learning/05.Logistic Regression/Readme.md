# Logistic Regression Classifier

An implementation and exploration of the Logistic Regression algorithm for binary classification, using Python. Primarily intended for learning purposes.

## Features

-   Outputs probability estimates for class membership.
-   Utilizes the sigmoid (logistic) function to map linear combinations to probabilities [0, 1].
-   Decision boundary visualization for 2D data.
-   Model performance evaluation (accuracy, confusion matrix, precision, recall, etc.).

## What is Logistic Regression?

Despite its name containing "Regression," Logistic Regression is a fundamental and widely used **linear model for classification** tasks, especially binary classification. It works by first calculating a weighted sum of the input features (plus a bias term), similar to linear regression. However, this linear combination is then passed through a **sigmoid (logistic) function**, which squashes the output into the range [0, 1]. This output is interpreted as the probability of the input belonging to the positive class (e.g., class '1'). A threshold (commonly 0.5) is applied to this probability to make the final class prediction (0 or 1).

## Getting Started

### Need

Make sure you have the following libraries installed:

```bash
scikit-learn
numpy
matplotlib
pandas
mlxtend.plotting 

```
### File structure and organization
```bash
README.md                          # This file
Mats.csv                      # Placeholder for the dataset used
logistic_regression.ipynb  # The main implimentation
*.png                              # Images used for illustrative purposes
```

Authored by `Mark Munyi`