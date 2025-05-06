# Decision Tree Classifier

Decision Tree algorithm, implemented using Python and scikit-learn.
## Features

-   Classification.
-   Tree structure visualization to understand decision rules.
-   Calculation of feature importances.
-   Configurable hyperparameters .
-   Model performance evaluation (accuracy, confusion matrix, precision, recall, F1-score).
-   Decision boundary visualization for 2D data.

## What is a Decision Tree?

A Decision Tree is a non-parametric supervised learning algorithm used for both classification and regression tasks. It works by learning simple decision rules inferred from the data features to predict the value of a target variable. The model is built by recursively partitioning the data into subsets based on the values of input features that best split the data according to a chosen criterion (e.g., Gini impurity or information gain/entropy).

The tree structure consists of:
-   **Root Node:** Represents the entire dataset.
-   **Internal Nodes:** Represent tests on input features.
-   **Branches:** Represent the outcomes of these tests.
-   **Leaf Nodes (Terminal Nodes):** Represent the class labels (for classification) or continuous values (for regression) assigned to instances that reach that leaf.

Decision Trees are popular due to their interpretability and ease of visualization.

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
### File Structure
```bash
.png  # images used in the explanation
.csv # file containing the 
.ipynb # main file with the implementation
README.md                         # This file
```

This uses a publicly available data set on Breat cancer diagnosis.