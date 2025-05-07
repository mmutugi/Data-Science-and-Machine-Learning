# Deep Neural Network (DNN) for Image Classification

An implementation and exploration of a Deep Neural Network for Image classification, built using Python from scratch and with TensorFlow. Designed for educational purposes.

## Features

-   Multi-layer neural network architecture (i.e., multiple hidden layers).
-   Training via **Backpropagation** using an optimizer.
-   Configurable network parameters: number of layers, neurons per layer, learning rate, epochs, batch size.
-   Performance evaluation using appropriate metrics (e.g., accuracy, ).


## What is a Deep Neural Network (DNN)?

A Deep Neural Network (DNN) is a type of Artificial Neural Network (ANN) with **multiple hidden layers** between the input and output layers. This "depth" allows DNNs to learn complex hierarchical feature representations from data. Each layer typically consists of several neurons, and each neuron in one layer is connected to neurons in the subsequent layer, with associated weights and biases.

During a forward pass, input data travels through the network, layer by layer, undergoing linear transformations (weighted sums) followed by non-linear activation functions. The final layer produces the output (e.g., class probabilities or a continuous value). The network "learns" by adjusting its weights and biases through an optimization process called **backpropagation**, which calculates the gradient of a loss function with respect to the network parameters and updates them to minimize this loss. DNNs are the foundation for many state-of-the-art results in fields like image recognition, natural language processing, and speech recognition.

## Getting Started

### Need

Make sure you have the following libraries and frameworks installed:

```bash
# Core Framework
Tensorflow


# Essential Libraries
numpy
matplotlib (for plotting loss, accuracy, etc.)
pandas (for data loading and manipulation)

(You can typically install these using pip: pip install tensorflow numpy matplotlib pandas )File Structure 
deep.ipnb   #main file with the implimentation
deep_Tensorflow   #implementation with TensorFlow
README.md                    # This file
*.png                        # Images used in the description

```
Authored by `Mark Munyi` for CMOR 438 taught by `Randy Davila, Ph.D.`