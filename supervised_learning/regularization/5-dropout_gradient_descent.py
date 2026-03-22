#!/usr/bin/env python3
"""
Contains the function dropout_gradient_descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.
    Args:
        Y: one-hot numpy.ndarray (classes, m) containing correct labels
        weights: dictionary of the weights and biases
        cache: dictionary of the outputs and dropout masks of each layer
        alpha: learning rate
        keep_prob: probability that a node will be kept
        L: number of layers of the network
    Returns: None (weights updated in place)
    """
    m = Y.shape[1]
    # Initialize backprop with the gradient of the loss w.r.t last layer (A_L)
    # For Softmax + Cross-Entropy, dZ is simply (A_L - Y)
    dZ = cache['A{}'.format(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A{}'.format(i - 1)]
        W = weights['W{}'.format(i)]

        # Calculate gradients for weights and biases
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            # Calculate dA for the previous layer
            dA = np.matmul(W.T, dZ)
            
            # Apply the dropout mask and scaling from the forward pass
            # This is critical for consistency
            dA = (dA * cache['D{}'.format(i - 1)]) / keep_prob
            
            # Calculate dZ for the hidden layer (derivative of tanh is 1 - A^2)
            dZ = dA * (1 - (A_prev ** 2))

        # Update weights and biases in place
        weights['W{}'.format(i)] -= alpha * dW
        weights['b{}'.format(i)] -= alpha * db
