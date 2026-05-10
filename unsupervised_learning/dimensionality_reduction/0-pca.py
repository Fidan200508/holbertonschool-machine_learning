#!/usr/bin/env python3
"""
Performs PCA on a dataset
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset to maintain a fraction of the variance
    Args:
        X: numpy.ndarray of shape (n, d) where all dimensions have 0 mean
        var: fraction of the variance that the PCA should maintain
    Returns:
        W: weights matrix of shape (d, nd)
    """
    # Singular Value Decomposition
    # X = U * S * Vh
    # vh contains the principal components (eigenvectors)
    _, s, vh = np.linalg.svd(X)

    # Calculate cumulative variance
    # Squared singular values are proportional to the variance
    squared_s = s ** 2
    cumulative_variance = np.cumsum(squared_s) / np.sum(squared_s)

    # Determine the number of dimensions needed to reach the threshold
    # np.where returns indices where condition is true; we take the first one
    nd = np.where(cumulative_variance >= var)[0][0] + 1

    # Extract the weights matrix W
    # The components are in the rows of vh, so we take the first nd rows
    # and transpose to get shape (d, nd)
    W = vh[:nd].T

    return W
