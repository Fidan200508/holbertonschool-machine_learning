#!/usr/bin/env python3
"""
Contains the function early_stopping
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early
    Args:
        cost: current validation cost of the neural network
        opt_cost: lowest recorded validation cost
        threshold: threshold used for early stopping
        patience: patience count used for early stopping
        count: count of how long the threshold has not been met
    Returns:
        A boolean of whether to stop early, and the updated count
    """
    # Check if the current cost has improved by more than the threshold
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1

    # Check if the counter has reached the patience limit
    if count >= patience:
        return True, count

    return False, count
