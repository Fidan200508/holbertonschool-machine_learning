#!/usr/bin/env python3
"""Bayesian Optimization module."""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(
        self,
        f,
        X_init,
        Y_init,
        bounds,
        ac_samples,
        l=1,
        sigma_f=1,
        xsi=0.01,
        minimize=True
    ):
        """Class constructor."""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculates the next best sample location."""
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            y_sample = np.min(self.gp.Y)
            improvement = y_sample - mu - self.xsi
        else:
            y_sample = np.max(self.gp.Y)
            improvement = mu - y_sample - self.xsi

        Z = np.zeros_like(mu)
        mask = sigma > 0
        Z[mask] = improvement[mask] / sigma[mask]

        EI = np.zeros_like(mu)
        EI[mask] = improvement[mask] * norm.cdf(Z[mask]) + \
            sigma[mask] * norm.pdf(Z[mask])

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI
