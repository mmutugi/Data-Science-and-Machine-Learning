# K-Nearest Neighbors (KNN) Classifier

A straightforward implementation of the K-Nearest Neighbors algorithm for classification, using Python and popular data science libraries. Built primarily for educational exploration.

## Features

-   Classification using the KNN algorithm (`scikit-learn` implementation).
-   Configurable number of neighbors (`k`).
-   Support for various distance metrics (Euclidean, Manhattan, etc.).
-   Decision boundary visualization for 2D data (using `mlxtend` or similar).
-   Model performance evaluation (e.g., accuracy, confusion matrix).

## What is KNN?

K-Nearest Neighbors (KNN) is a simple, instance-based supervised learning algorithm often used for classification (and regression). For a given new data point, it finds the 'k' closest data points in the training set based on a distance metric. The prediction for the new point is determined by the majority class among these 'k' neighbors. It's considered a "lazy learner" because it doesn't build a general internal model but stores the entire training dataset for prediction.

## Getting Started

### Need

Make sure you have the following libraries installed:

```bash
scikit-learn
numpy
matplotlib
pandas
mlxtend.plotting (Optional, for decision boundary plots)

```

### File structure

- knearest.ipynb     # Jupyter Notebook with KNN implementation and analysis
- Readme.md              # This file
- data.csv          # Placeholder for the dataset used
- *.png                  # Images used in the demonstration

In this book, I use a Breast Cancer dataset that is publicly available on the web and kaggle.

Authored by Mark Munyi for CMOR 438 taught by Randy R. Davila, Ph.D