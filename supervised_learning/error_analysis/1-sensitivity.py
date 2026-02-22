#!/usr/bin/env python3
"""1-sensitivity.py"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity (recall) for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                                   Rows = true labels, Columns = predicted labels

    Returns:
        numpy.ndarray: Sensitivity for each class, shape (classes,)
                       Calculated as TP / (TP + FN)
    """
    # True positives for each class are the diagonal elements
    true_positives = np.diag(confusion)

    # Total actual positives per class is the sum of each row
    actual_positives = np.sum(confusion, axis=1)

    # Sensitivity = TP / (TP + FN)
    sens = true_positives / actual_positives

    return sens
