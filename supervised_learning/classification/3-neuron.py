#!/usr/bin/env python3
"""Single neuron performing binary classification."""

import numpy as np


class Neuron:
    """Defines a single neuron for binary classification."""

    def __init__(self, nx):
        """Initialize the neuron."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Return the weights vector."""
        return self.__W

    @property
    def b(self):
        """Return the bias."""
        return self.__b

    @property
    def A(self):
        """Return the activated output."""
        return self.__A

    def forward_prop(self, X):
        """Calculate forward propagation."""
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))

        return self.__A

    def cost(self, Y, A):
        """Calculate logistic regression cost."""
        m = Y.shape[1]

        cost = -(
            1 / m
        ) * np.sum(
            Y * np.log(A)
            + (1 - Y) * np.log(1.0000001 - A)
        )

        return cost
