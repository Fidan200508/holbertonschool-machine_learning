#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification
"""
import numpy as np


class Neuron:
    """
    Represents a single neuron performing binary classification
    """
    def __init__(self, nx):
        """
        Initializes the neuron
        Args:
            nx: number of input features
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
        Returns:
            The private attribute __A
        """
        # Linear transform: Z = WX + b
        # Using np.matmul for matrix multiplication
        z = np.matmul(self.__W, X) + self.__b

        # Activation: Sigmoid function
        self.__A = 1 / (1 + np.exp(-z))

        return self.__A
