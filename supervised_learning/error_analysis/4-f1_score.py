#!/usr/bin/env python3
"""4-f1_score.py"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                                   Rows = true labels, Columns = predicted
                                   labels

    Returns:
        numpy.ndarray: F1 score for each class, shape (classes,)
                       Calculated as 2 * (precision * sensitivity) / 
                       (precision + sensitivity)
    """
    rec = sensitivity(confusion)
    prec = precision(confusion)
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1
