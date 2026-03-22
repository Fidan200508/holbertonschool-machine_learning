#!/usr/bin/env python3
"""Gradient descent with L2 regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization.

    Parameters:
    Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
    weights: dictionary of weights and biases
    cache: dictionary of layer outputs
    alpha: learning rate
    lambtha: L2 regularization parameter
    L: number of layers
    """

    m = Y.shape[1]

    # Initialize dZ as difference for last layer
    dZ = cache["A{}".format(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache["A{}".format(i - 1)]
        W = weights["W{}".format(i)]

        # Compute gradients
        dW = (np.matmul(dZ, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Update weights and biases
        weights["W{}".format(i)] -= alpha * dW
        weights["b{}".format(i)] -= alpha * db

        if i > 1:
            # Backpropagate dZ to previous layer using tanh derivative
            dZ = np.matmul(W.T, dZ) * (1 - np.power(A_prev, 2))
