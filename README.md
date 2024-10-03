# neural::network

**neural::network** is a lightweight C++ library designed for the creation, training, and evaluation of neural networks. It provides essential components necessary for building fully connected neural networks, including matrix operations, activation functions, feedforward logic, backpropagation mechanisms, and mutation functionality.

## Table of Contents

1. [Overview](#overview)
2. [Creating a Neural Network](#creating-a-neural-network)
3. [Feedforward Pass](#feedforward-pass)
   - [Feedforward Functions](#feedforward-functions)
4. [Backpropagation](#backpropagation)
   - [Backpropagation Functions](#backpropagation-functions)
5. [Matrix Operations](#matrix-operations)
6. [Mutation](#mutation)
7. [Usage Example](#usage-example)
8. [License](#license)
9. [Acknowledgment](#acknowledgment)

## Overview

The **neural::network** library encompasses the following primary functionalities:

- **Matrix Operations**: Basic operations including multiplication, addition, subtraction, and element-wise operations.
- **Activation Functions**: Common activation functions such as Sigmoid, Softmax, and ReLU.
- **Feedforward Logic**: Methods to compute outputs through the network.
- **Backpropagation**: Mechanisms to calculate gradients and update weights and biases based on the error.
- **Mutation**: Introduce randomness to the weights and biases for evolutionary algorithms.

## Creating a Neural Network

To create a neural network, define a `network` structure and initialize it with necessary parameters.

### Example:

```cpp
network myNetwork;
myNetwork.input = {{0.0, 1.0}, {1.0, 0.0}}; // Example input
// Add layers to your network as needed
```

### Adding Layers

Layers can be added to your network. Each layer contains weights and biases that will be updated during training.

## Feedforward Pass

The library provides several methods for performing the feedforward pass through the network.

### Feedforward Functions

1. **`feedforward(network *network_)`**: 
   - This function computes the final output of the network based on the input data stored within the `network` structure.
   - **Returns**: A vector of the final outputs from the network.

   **Example**:
   ```cpp
   std::vector<std::vector<float>> output = feedforward(&myNetwork);
   ```

2. **`feedforward_a(network *network_)`**: 
   - Computes the activations for each layer during the feedforward pass and returns these activations.
   - **Returns**: A vector containing the activations of all layers, including the input layer.

   **Example**:
   ```cpp
   std::vector<std::vector<std::vector<float>>> activations = feedforward_a(&myNetwork);
   ```

3. **`feedforward_z(network *network_)`**: 
   - Similar to `feedforward_a`, but returns the pre-activation values (z) for each layer.
   - **Returns**: A vector containing the pre-activations for all layers, including the input layer.

   **Example**:
   ```cpp
   std::vector<std::vector<std::vector<float>>> preActivations = feedforward_z(&myNetwork);
   ```

4. **`feedforward(std::vector<std::vector<float>> input_alt, network *network_)`**: 
   - Computes the final output using an alternative input provided by the user.
   - **Parameters**:
     - `input_alt`: The input data for the feedforward pass.
     - `network_`: The neural network being used.
   - **Returns**: A vector of the final outputs from the network.

   **Example**:
   ```cpp
   std::vector<std::vector<float>> output = feedforward({{0.0, 1.0}}, &myNetwork);
   ```

5. **`feedforward_a(std::vector<std::vector<float>> input_alt, network *network_)`**: 
   - Similar to `feedforward_a`, but uses the provided input data for computations.
   - **Returns**: A vector containing the activations of all layers, including the input layer.

   **Example**:
   ```cpp
   std::vector<std::vector<std::vector<float>>> activations = feedforward_a({{0.0, 1.0}}, &myNetwork);
   ```

6. **`feedforward_z(std::vector<std::vector<float>> input_alt, network *network_)`**: 
   - Similar to `feedforward_z`, but uses the provided input data for computations.
   - **Returns**: A vector containing the pre-activations for all layers, including the input layer.

   **Example**:
   ```cpp
   std::vector<std::vector<std::vector<float>>> preActivations = feedforward_z({{0.0, 1.0}}, &myNetwork);
   ```

## Backpropagation

Backpropagation is a critical component for training the neural network, allowing it to learn from errors.

### Backpropagation Functions

1. **`backpropagate(network *network_, std::vector<std::vector<float>> output, std::vector<std::vector<float>> target, float _n_)`**: 
   - Updates the weights and biases of the network based on the output and target values.
   - **Parameters**:
     - `output`: The output obtained from the network.
     - `target`: The expected output (ground truth) for the training sample.
     - `_n_`: The learning rate for updating the weights.
   - **Returns**: The updated network.

   **Example**:
   ```cpp
   std::vector<std::vector<float>> target = {{1.0}, {0.0}}; // Example target
   myNetwork = backpropagate(&myNetwork, output, target, 0.01); // Learning rate set to 0.01
   ```

2. **`backpropagate_batch(network *network_, std::vector<std::vector<std::vector<float>>> batch_inputs, std::vector<std::vector<std::vector<float>>> batch_targets, float _n_)`**: 
   - Performs backpropagation for a batch of inputs and targets, averaging the gradients.
   - **Parameters**:
     - `batch_inputs`: A vector of input vectors for the batch.
     - `batch_targets`: A vector of target vectors for the batch.
     - `_n_`: The learning rate for updating the weights.
   - **Returns**: The updated network.

   **Example**:
   ```cpp
   std::vector<std::vector<std::vector<float>>> batchInputs = {{{0.0, 1.0}}, {{1.0, 0.0}}}; // Example batch inputs
   std::vector<std::vector<std::vector<float>>> batchTargets = {{{1.0}}, {{0.0}}}; // Example batch targets
   myNetwork = backpropagate_batch(&myNetwork, batchInputs, batchTargets, 0.01); // Learning rate set to 0.01
   ```

## Matrix Operations

The library includes essential matrix operations encapsulated within the `matrix` structure. Here are the primary functions you can use:

1. **Matrix Multiplication**: 
   - **Function**: `matrix.multiply(matrix1, matrix2)`
   - **Description**: Multiplies two matrices.
   
   **Example**:
   ```cpp
   std::vector<std::vector<float>> result = matrix.multiply(matrixA, matrixB);
   ```

2. **Element-wise Addition**: 
   - **Function**: `matrix.add(matrix1, matrix2)`
   - **Description**: Adds two matrices element-wise.
   
   **Example**:
   ```cpp
   std::vector<std::vector<float>> result = matrix.add(matrixA, matrixB);
   ```

3. **Element-wise Subtraction**: 
   - **Function**: `matrix.difference(matrix1, matrix2)`
   - **Description**: Subtracts the second matrix from the first element-wise.
   
   **Example**:
   ```cpp
   std::vector<std::vector<float>> result = matrix.difference(matrixA, matrixB);
   ```

4. **Element-wise Multiplication (Hadamard)**: 
   - **Function**: `matrix.multiply_hadamard(matrix1, matrix2)`
   - **Description**: Performs element-wise multiplication of two matrices.
   
   **Example**:
   ```cpp
   std::vector<std::vector<float>> result = matrix.multiply_hadamard(matrixA, matrixB);
   ```

5. **Transposing a Matrix**: 
   - **Function**: `matrix.transpose(matrix)`
   - **Description**: Transposes the given matrix.
   
   **Example**:
   ```cpp
   std::vector<std::vector<float>> transposed = matrix.transpose(matrixA);
   ```

6. **Creating Zero Matrices**: 
   - **Function**: `matrix.zeros(matrix)`
   - **Description**: Creates a matrix of zeros with the same dimensions as the given matrix.
   
   **Example**:
   ```cpp
   std::vector<std::vector<float>> zeroMatrix = matrix.zeros(matrixA);
   ```

7. **Summing Matrices**: 
   - **Function**: `matrix.sum(matrix1, matrix2)`
   - **Description**: Sums two matrices element-wise.
   
   **Example**:
   ```cpp
   std::vector<std::vector<float>> result = matrix.sum(matrixA, matrixB);
   ```

## Mutation

The **neural::network** library includes functionality for mutation, allowing for variations in weights and biases, which can be beneficial in evolutionary algorithms.

1. **Weight Mutation**:
   - **Function**: `mutate_weights(network *network_, float mutation_rate)`
   - **Description**: Mutates the weights in the network based on a specified mutation rate.
   - **Parameters**:
     - `mutation_rate`: A float value representing the likelihood of each weight being mutated.

   **Example**:
   ```cpp
   mutate_weights(&myNetwork, 0.1); // 10% mutation rate
   ```

2. **Bias Mutation**:
   - **Function**: `mutate_biases(network *network_, float mutation_rate)`
   - **Description**: Mutates the biases in the network based on a specified mutation rate.
   - **Parameters**:
     - `mutation_rate`: A float value representing the likelihood of each bias being mutated.

   **Example**:
   ```cpp
   mutate_biases(&myNetwork, 0.1); // 10% mutation rate
   ```

## Usage Example

Here is a simple example of using the **neural::network** library:

```cpp
#include "neural_network.h" // Include your header file

int main() {
    // Initialize the network
    network myNetwork;

    // Setup input, layers, and other parameters...
    
    // Perform a feedforward pass
    std::vector<std::vector<float>> output = feedforward(&myNetwork);

    // Define targets and perform backpropagation
    std::vector<std::vector<float>> target = {{1.0}, {0.0}};
    myNetwork = backpropagate(&myNetwork, output, target, 0.01);

    // Perform mutations if necessary
    mutate_weights(&myNetwork, 0.1);
    mutate_biases(&myNetwork, 0.1);

    return 0;
}
```

## License

This library is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgment

This project is inspired by the principles of neural networks and aims to provide a foundational tool for learning and experimentation in machine learning and artificial intelligence.

## Disclaimer

This documentation was written by artificial intelligence, please excuse us for any inaccuracies. I plan to rewrite this documentation in the future. Look out for any updates.
