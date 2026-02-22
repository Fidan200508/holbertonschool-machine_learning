#!/usr/bin/env python3
"""0-create_confusion.py"""
import numpy as np

def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Args:
        labels (numpy.ndarray): One-hot array of correct labels, shape (m, classes)
        logits (numpy.ndarray): One-hot array of predicted labels, shape (m, classes)

    Returns:
        numpy.ndarray: Confusion matrix of shape (classes, classes) with dtype=float
                       Rows = true labels, Columns = predicted labels
    """
    m, classes = labels.shape

    # Initialize confusion matrix with zeros, dtype=float
    confusion = np.zeros((classes, classes), dtype=float)

    # Convert one-hot to integer class labels
    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(logits, axis=1)

    # Fill confusion matrix
    for t, p in zip(true_labels, pred_labels):
        confusion[t, p] += 1

    return confusion
