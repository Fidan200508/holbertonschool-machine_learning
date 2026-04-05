#!/usr/bin/env python3
"""
Contains the function l2_reg_gradient_descent
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases of a neural network using gradient
    descent with L2 regularization
    Args:
        Y: one-hot numpy.ndarray (classes, m) containing correct labels
        weights: dictionary of weights and biases
        cache: dictionary of the outputs of each layer
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers of the network
    Returns: None (updated in place)
    """
    m = Y.shape[1]
    # dZ for the last layer (Softmax + Cross-Entropy)
    dZ = cache['A{}'.format(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A{}'.format(i - 1)]
        W_key = 'W{}'.format(i)
        b_key = 'b{}'.format(i)
        W = weights[W_key]

        # Calculate dW with L2 regularization term: (lambda/m) * W
        dW = (1 / m) * (np.matmul(dZ, A_prev.T) + (lambtha * W))
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            # Backpropagate to the previous layer
            # Derivative of tanh: 1 - A^2
            dZ = np.matmul(W.T, dZ) * (1 - (A_prev ** 2))

        # Update weights and biases in place
        weights[W_key] -= alpha * dW
        weights[b_key] -= alpha * db
