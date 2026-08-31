#!/usr/bin/env python3
"""Neural network with one hidden layer."""

import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer."""

    def __init__(self, nx, nodes):
        """Initialize the neural network."""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")

        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Return the weights of the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """Return the bias of the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """Return the activated output of the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """Return the weights of the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """Return the bias of the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """Return the activated output of the output neuron."""
        return self.__A2
