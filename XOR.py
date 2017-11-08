""" Neural Network XOR Problem """

import numpy as np


def sigmoid(g):
    return 1 / (1 + np.exp(-2 * g))


def sigmoid_gradient(g):
    return g * (1 - g)


def feedForwardProp(input_layer, output_layer, hidden_weights, output_weights, bias):
    z2 = np.dot(input_layer, hidden_weights)
    a2 = sigmoid(z2)
    a2 = a2.T
    a2 = np.vstack((a2, bias)).T
    z3 = np.dot(a2, output_weights)
    a3 = sigmoid(z3)
    return a2, a3, hidden_weights, output_weights


def backPropogation(input_layer, output_layer, hidden_weights, output_weights, bias, iterations):
    for _ in range(iterations):
        a2, a3, hidden_weights, output_weights = feedForwardProp(
            input_layer, output_layer, hidden_weights, output_weights, bias)

        error_a3 = output_layer - a3
        error_a2 = np.dot(error_a3, output_weights[0:2, :].T) * \
            sigmoid(np.dot(input_layer, hidden_weights))

        delta_a3 = error_a3 * sigmoid_gradient(a3)
        delta_a2 = error_a2 * sigmoid_gradient(a2[:, 0:2])

        # Update weights
        output_weights += np.dot(a2.T, delta_a3)
        hidden_weights += np.dot(input_layer.T, delta_a2)

    return a3


# Data
input_layer = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
output_layer = np.array([[0, 1, 1, 0]]).T

# Randomly initialising weights
np.random.seed(1)
hidden_weights = np.random.random((3, 2))
output_weights = np.random.random((3, 1))

# Number of iterations
iterations = 10000

# Bias term
bias = np.ones((1, 4))

print(backPropogation(input_layer, output_layer, hidden_weights, output_weights, bias, iterations))
