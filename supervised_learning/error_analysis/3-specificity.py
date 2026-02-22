#!/usr/bin/env python3
"""3-specificity.py"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                                   Rows = true labels, Columns = predicted
                                   labels

    Returns:
        numpy.ndarray: Specificity for each class, shape (classes,)
                       Calculated as TN / (TN + FP)
    """
    classes = confusion.shape[0]
    spec = np.zeros(classes, dtype=float)

    # Total sum of all elements
    total = np.sum(confusion)

    for i in range(classes):
        TP = confusion[i, i]
        FP = np.sum(confusion[:, i]) - TP
        FN = np.sum(confusion[i, :]) - TP
        TN = total - (TP + FP + FN)
        spec[i] = TN / (TN + FP)

    return spec
