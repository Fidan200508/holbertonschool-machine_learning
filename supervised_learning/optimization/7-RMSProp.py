#!/usr/bin/env python3
"""Updates variables using the RMSProp optimization algorithm."""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Update a variable using RMSProp optimization."""
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    var_new = var - alpha * grad / (
        np.sqrt(s_new) + epsilon
    )

    return var_new, s_new
