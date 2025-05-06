#  Machine Learning Python Package

A Python package demonstrating:

- **Perceptron Classification** — Training, prediction & visualization of decision boundaries.
- **Logistic Regression with SGD** — Binary classification using MSE or cross-entropy loss.
- **Visualization Tools** — Plotting decision boundaries, weight magnitudes & logistic S-curves.

---

## Description

This project provides a hands-on implementation of two foundational machine learning algorithms: **Perceptron** and **Logistic Regression**.

It is designed impliment a sample of:

- Binary classification principles
- Weight and bias updates during training
- Performance evaluation (accuracy, loss)
- Visualization of model behavior

The package includes reusable classes, visualization methods, and example notebooks for applying these models to real or synthetic datasets.

---

## Project Structure 

```yaml
Package/
├── example_usage/
│   └── usage.ipynb               # Example notebook demonstrating package usage
├── ml_packages/
│   ├── perceptron.py             # Perceptron class with training, predict & visualization
│   └── logistic_regression.py # Logistic Regression class with SGD & visualization
├── tests/
│   ├── test_perceptron.py        # Unit tests for Perceptron
│   └── test_logistic_regression.py # Unit tests for Logistic Regression
├── requirements.txt              # Project dependencies
├── README.md                     # This documentation
├── setup.py                      # Setup script for pip install
└── .gitignore                    # Excludes unnecessary files from git
```

### Installation
git clone <this repository>
cd Package
pip install -e . or pip install ml-packages

### Requires
numpy
matplotlib
scikit-learn
pytest

### Authored:
Mark Munyi

