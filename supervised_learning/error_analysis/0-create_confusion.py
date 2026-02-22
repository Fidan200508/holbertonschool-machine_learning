#!/usr/bin/env python3
"""
0. Create Confusion
Module that contains a function to create a confusion matrix
"""

import numpy as np

def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Parameters:
    - labels: numpy.ndarray of shape (m, classes) containing the correct labels (one-hot)
    - logits: numpy.ndarray of shape (m, classes) containing the predicted labels (one-hot)

    Returns:
    - confusion: numpy.ndarray of shape (classes, classes)
      Rows represent true labels, columns represent predicted labels
    """
    # Convert one-hot encoded labels to class indices
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)

    # Number of classes
    classes = labels.shape[1]

    # Initialize confusion matrix with zeros
    confusion = np.zeros((classes, classes), dtype=int)

    # Fill confusion matrix
    for t, p in zip(true_classes, pred_classes):
        confusion[t, p] += 1

    return confusion
