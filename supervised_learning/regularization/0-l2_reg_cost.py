#!/usr/bin/env python3
"""L2 Regularization Cost"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    Parameters:
    cost: original cost (without regularization)
    lambtha: regularization parameter
    weights: dictionary of weights and biases
    L: number of layers
    m: number of data points

    Returns:
    cost with L2 regularization
    """

    l2_sum = 0

    # Sum of squared weights
    for i in range(1, L + 1):
        W = weights["W{}".format(i)]
        l2_sum += np.sum(np.square(W))

    # L2 regularization term
    l2_term = (lambtha / (2 * m)) * l2_sum

    return cost + l2_term
