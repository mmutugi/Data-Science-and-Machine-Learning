# Label Propagation for Semi-Supervised Learning

An exploration of the Label Propagation algorithm for semi-supervised classification, implemented using Python and scikit-learn. Built for educational and practical demonstration purposes.

## Features

-   Semi-supervised classification (learns from a mix of labeled and unlabeled data).
-   Utilizes a graph-based approach where labels "propagate" from labeled to unlabeled nodes.
-   Model performance evaluation (accuracy, confusion matrix).

## What is Label Propagation?

Label Propagation is a **semi-supervised learning algorithm** that assigns labels to previously unlabeled data points based on their similarity or proximity to labeled data points. It constructs a graph where nodes are data points and edges represent the similarity between them.

The core idea is that labels "propagate" from the initially labeled nodes to their neighbors, and then to their neighbors' neighbors, and so on, across the graph. Each node iteratively updates its label distribution based on the label distributions of its neighbors until a global convergence is reached. Nodes with strong connections to a particular labeled class are likely to adopt that class label. This allows the model to leverage the structure of both labeled and unlabeled data to make predictions.

## Getting Started

### Need

Make sure you have the following libraries installed:

```bash
scikit-learn
numpy
matplotlib (for visualization, if applicable)
pandas (for data handling)


.ipynb                       # Main file with the implementation
README.md                            # This file
.png #images used in the description
.csv # csv files created by the implentation of the algorithm. 
```
Authored by Mark Munyi for CMOR 438 taught by Randy Davila