#!/usr/bin/env python3
"""
Contains the function dropout_forward_prop
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout
    Args:
        X: numpy.ndarray (nx, m) containing the input data
        weights: dictionary of the weights and biases
        L: number of layers in the network
        keep_prob: probability that a node will be kept
    Returns:
        cache: dictionary containing outputs and dropout masks
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W{}'.format(i)]
        b = weights['b{}'.format(i)]
        A_prev = cache['A{}'.format(i - 1)]

        # Linear Step: Z = W * A_prev + b
        Z = np.matmul(W, A_prev) + b

        if i == L:
            # Last layer: Softmax activation
            t = np.exp(Z)
            cache['A{}'.format(i)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            # Hidden layers: Tanh activation
            A = np.tanh(Z)

            # Create dropout mask
            # np.random.rand creates a matrix of same shape as A
            # comparing to keep_prob creates a boolean mask (converted to int)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob).astype(int)

            # Apply mask and scale (Inverted Dropout)
            A = (A * D) / keep_prob

            cache['D{}'.format(i)] = D
            cache['A{}'.format(i)] = A

    return cache
