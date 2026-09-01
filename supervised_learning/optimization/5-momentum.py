#!/usr/bin/env python3
"""Updates variables using gradient descent with momentum."""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update a variable using momentum optimization."""
    v_new = beta1 * v + (1 - beta1) * grad
    var_new = var - alpha * v_new

    return var_new, v_new
