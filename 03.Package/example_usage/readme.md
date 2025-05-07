## Brief Overview: Perceptron vs. Logistic Regression Script

This Python script demonstrates a comparative analysis between the Perceptron algorithm and Logistic Regression (using two different loss functions) for a binary classification task.

**Key Steps in the Script:**

**To run the tests, just do pytest on the terminal**

1.  **Setup & Data Preparation:**
    * Imports necessary libraries (`numpy`, `matplotlib`, `sklearn`).
    * Assumes custom `Perceptron` and `LogisticRegression` classes are available from an `ml_packages` module.
    * Loads the Iris dataset.
    * Simplifies the dataset for binary classification by selecting only the first two Iris species (Setosa and Versicolor) and using only the first two features (e.g., sepal length and width) for easier 2D visualization.
    * Splits this modified data into training and testing sets.

2.  **Perceptron Model:**
    * Instantiates a `Perceptron` classifier.
    * Trains the Perceptron on the training data.
    * Makes predictions on the test set.
    * Calculates and prints the accuracy of the Perceptron model.
    * Includes a call to a  method `plot_decision_boundary` to visualize the Perceptron's learned boundary.

3.  **Logistic Regression Model:**
    * **With MSE Loss:**
        * Instantiates a `LogisticRegression` model.
        * Trains it using Mean Squared Error (MSE) as the loss function.
        * Predicts on the test set and prints its accuracy.
        * Calls a method `plot_cost_function` to visualize its training cost.
    * **With Cross-Entropy Loss:**
        * Instantiates another `LogisticRegression` model.
        * Trains it using Binary Cross-Entropy as the loss function (more standard for classification).
        * Predicts on the test set and prints its accuracy.
        * Calls `plot_cost_function` to visualize its training cost.

In essence, the script sets up a simple binary classification problem, trains three different model configurations (Perceptron, Logistic Regression with MSE, Logistic Regression with Cross-Entropy), evaluates their accuracies, and prepares for visualizing their decision boundaries and learning curves.
