#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases of a neural network using gradient descent with L2 regularization.

    Parameters:
    Y: numpy.ndarray of shape (classes, m), one-hot labels
    weights: dictionary of weights and biases
    cache: dictionary of layer outputs
    alpha: learning rate
    lambtha: L2 regularization parameter
    L: number of layers
    """

    m = Y.shape[1]  # number of data points
    dZ = cache["A{}".format(L)] - Y  # last layer (softmax) gradient

    for l in reversed(range(1, L + 1)):
        A_prev = cache["A{}".format(l - 1)]
        W = weights["W{}".format(l)]

        # Gradient of weights with L2
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W

        # Gradient of biases
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases
        weights["W{}".format(l)] -= alpha * dW
        weights["b{}".format(l)] -= alpha * db

        # Compute dZ for next layer if not first layer
        if l > 1:
            A_prev_linear = cache["A{}".format(l - 1)]
            dZ = np.matmul(W.T, dZ) * (1 - A_prev_linear**2)  # tanh derivative
