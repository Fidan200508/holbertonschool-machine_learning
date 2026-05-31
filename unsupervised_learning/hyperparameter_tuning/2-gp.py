#!/usr/bin/env python3
"""Gaussian Process module."""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Class constructor."""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix."""
        a = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        b = np.sum(X2 ** 2, axis=1)
        c = 2 * np.matmul(X1, X2.T)

        sqdist = a + b - c

        return (self.sigma_f ** 2) * np.exp(
            -0.5 * sqdist / (self.l ** 2)
        )

    def predict(self, X_s):
        """Predicts the mean and variance of points."""
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        K_inv = np.linalg.inv(self.K)

        mu = np.matmul(np.matmul(K_s.T, K_inv), self.Y)
        cov = K_ss - np.matmul(np.matmul(K_s.T, K_inv), K_s)

        return mu.reshape(-1), np.diag(cov)

    def update(self, X_new, Y_new):
        """Updates the Gaussian Process with a new sample point."""
        X_new = X_new.reshape(1, 1)
        Y_new = Y_new.reshape(1, 1)

        self.X = np.concatenate((self.X, X_new), axis=0)
        self.Y = np.concatenate((self.Y, Y_new), axis=0)
        self.K = self.kernel(self.X, self.X)
