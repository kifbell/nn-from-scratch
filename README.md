# Neural Network from Scratch

This repository contains a C++ implementation of a fully-functional neural network library, developed as part of a programming project for the Bachelor’s Programme “HSE and University of London Double Degree Programme in Data Science and Business Analytics” at the Higher School of Economics.

## Project Overview

The primary goal of this project is to demystify the theoretical principles of neural networks and implement a robust and efficient, fully-functional neural network library in C++ from scratch. The library provides users with an interface to construct, train, and infer fully connected neural networks.

## Features

- Support for various types of layers including linear layers and activation functions.
- Multiple activation functions: ReLU, Sigmoid, Tanh, and Softmax.
- Loss functions: Cross-Entropy and Mean Squared Error.
- Optimizers: Stochastic Gradient Descent (SGD), Momentum, and AMSGrad.
- Data handling capabilities including reading from CSV files and batching.
- Extensive use of modern C++ features for performance and maintainability.

## Dataset

The MNIST dataset of handwritten digits was used for testing the performance of the neural network library. The dataset can be found [here](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Install the necessary dependencies:
    - C++23
    - Eigen (Version 3.10.0)
    - CMake (Version 3.28.1)
    - Git
    - clang-format
    - clang-tidy

3. Build the project:
    ```sh
    mkdir build
    cd build
    cmake ..
    make
    ```

## Usage

1. Include the neural network library in your project.
2. Create and configure the neural network using the provided classes and methods.
3. Use Library for yout own project (HFT)

### Example

Here is a simple example of how to use the library:

```cpp
#include <iostream>
#include "NeuralNetwork.h"
#include "Loss.h"
#include "Optimizer.h"

int main() {
    NeuralNet::NeuralNetwork nn;
    
    // Add layers
    nn.addLayer(new NeuralNet::LinearLayer(784, 128));
    nn.addLayer(new NeuralNet::ReLUActivation());
    nn.addLayer(new NeuralNet::LinearLayer(128, 10));
    nn.addLayer(new NeuralNet::SoftmaxActivation());

    // Load dataset
    NeuralNet::DataProvider dataProvider("path_to_mnist_dataset.csv");

    // Initialize loss and optimizer
    NeuralNet::CrossEntropyLoss lossFunction;
    NeuralNet::SGDOptimizer optimizer(nn, 0.01);

    // Train the network
    nn.train(dataProvider, lossFunction, optimizer, 10, 32);

    // Evaluate the network
    double accuracy = nn.evaluate(dataProvider, lossFunction);
    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}
