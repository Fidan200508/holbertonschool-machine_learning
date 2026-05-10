#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification
"""
import numpy as np


class Neuron:
    """
    Represents a single neuron
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

        # Weights initialized using a random normal distribution
        # Shape is (1, nx) to allow vectorization with input X
        self.W = np.random.randn(1, nx)
        # Bias initialized to 0
        self.b = 0
        # Activated output initialized to 0
        self.A = 0
