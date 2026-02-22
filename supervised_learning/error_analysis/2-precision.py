#!/usr/bin/env python3
"""2-precision.py"""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                                   Rows = true labels, Columns = predicted
                                   labels

    Returns:
        numpy.ndarray: Precision for each class, shape (classes,)
                       Calculated as TP / (TP + FP)
    """
    # True positives for each class are the diagonal elements
    true_positives = np.diag(confusion)

    # Total predicted positives per class is the sum of each column
    predicted_positives = np.sum(confusion, axis=0)

    # Precision = TP / (TP + FP)
    prec = true_positives / predicted_positives

    return prec
