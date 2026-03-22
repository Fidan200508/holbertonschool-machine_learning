#!/usr/bin/env python3
"""Gradient descent with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization.

    Y: one-hot numpy.ndarray of shape (classes, m)
    weights: dictionary of weights and biases
    cache: dictionary of outputs of each layer
    alpha: learning rate
    lambtha: L2 regularization parameter
    L: number of layers in the network
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y

    for l in reversed(range(1, L + 1)):
        A_prev = cache[f"A{l-1}"]
        W = weights[f"W{l}"]

        # Gradient of weights with L2 regularization
        dW = (np.matmul(dZ, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Update weights and biases
        weights[f"W{l}"] -= alpha * dW
        weights[f"b{l}"] -= alpha * db

        # Compute dZ for previous layer if not input
        if l > 1:
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - np.power(A_prev, 2))  # tanh derivative
