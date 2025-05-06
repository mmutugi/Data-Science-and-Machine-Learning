# Gradient Descent Classifier

A simple, customizable Gradient Descent algorithm for binary classification, built from scratch using Python and NumPy. Built for educational purposes only.

## Features
- Binary classification using gradient descent optimization
- Configurable learning rate and number of epochs
- Built-in training progress bar
- Decision boundary visualization for 2D data
- Model performance evaluation with percentages

## What is Gradient Descent?
Gradient Descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of the steepest descent as defined by the negative of the gradient. In machine learning, it's often used to minimize a loss function and optimize model weights. This algorithm simulates the process of learning through trial and error by minimizing error over time.

## Getting Started

### Need
Make sure you have the following libraries installed:
```bash
Numpy
Matplotlib
Tqdm
Pandas
mlxtend.plotting
```

## File Structure
```
gradient_descent.ipynb     # The main Gradient Descent class/file
README.md                  # This file
Matatu_data.csv            # The data set used for the training
*.png                      # Various images used for illustrations in this directory.
```

## Notes
- The model only supports **binary classification** with labels `0` and `1`.
- Works best with 2D input data for visualization.
- Can be enhanced with modifications like multi-class support, learning rate schedules, or regularization.

## Dataset Used
Check out the `matatu_data.csv` for a real-world inspired binary classification task. This is a synthetic dataset I personally created to model the Matatus in Kenya as shown in the `ipynb` file.

---
Authored by `Mark Munyi` for CMOR 438 taught by `Randy R. Davila, Ph.D.`


