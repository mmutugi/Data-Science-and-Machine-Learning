# Perceptron Classifier

A simple, customizable Perceptron algorithm for binary classification, built from scratch using Python and NumPy. Built for educational purposes only.

##  Features
- Binary classification using Perceptron learning rule
- Configurable learning rate and number of epochs
- Built-in training progress bar
- Decision boundary visualization for 2D data
- Model performance evaluation with percentages

##  What is a Perceptron?
The Perceptron is one of the earliest and simplest types of artificial neural networks. It’s a linear classifier that updates its weights based on the error between predicted and actual class labels. It is simply designed to mimmic the human brain.

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
perceptron.ipynb     # The main Perceptron class/file
README.md                   # This file
Matatu_data.csv      # The data set used for the training
*.png                # Various images used for illustrations in this directory.
```

## Notes
- The model only supports **binary classification** with labels `-1` and `1`.
- Works best with 2D input data for visualization.
- Can be enhanced with modificationa like multi-class support or different activation functions.

## Dataset used 
Check out the `matatu_data.csv` for a real-world inspired binary classification task.

---
### Authored by Mark Munyi for CMOR 438 taught by Randy R. Davila, Ph.D.

