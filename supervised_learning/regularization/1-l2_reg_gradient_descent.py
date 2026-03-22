#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization.

    Parameters:
    Y: numpy.ndarray of shape (classes, m), one-hot labels
    weights: dictionary of weights and biases
    cache: dictionary of layer outputs
    alpha: learning rate
    lambtha: L2 regularization parameter
    L: number of layers
    """
    m = Y.shape[1]

    # Initialize dZ for the output layer
    A_final = cache['A{}'.format(L)]
    dZ = A_final - Y

    for l in reversed(range(1, L + 1)):
        A_prev = cache['A{}'.format(l - 1)]
        W = weights['W{}'.format(l)]

        # Compute gradients with L2 regularization
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases
        weights['W{}'.format(l)] -= alpha * dW
        weights['b{}'.format(l)] -= alpha * db

        if l > 1:
            # Backprop through tanh activation
            dA_prev = np.matmul(W.T, dZ)
            A_prev_layer = cache['A{}'.format(l - 1)]
            dZ = dA_prev * (1 - A_prev_layer ** 2)
