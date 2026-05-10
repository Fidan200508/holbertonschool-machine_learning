#!/usr/bin/env python3
"""
Calculates the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    Determines the definiteness of a numpy.ndarray
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if matrix is square and not empty
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1] \
       or matrix.size == 0:
        return None

    # Check if the matrix is symmetric
    # (Definiteness is typically defined for symmetric matrices)
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        eigenvalues = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None

    pos = np.all(eigenvalues > 0)
    pos_semi = np.all(eigenvalues >= 0)
    neg = np.all(eigenvalues < 0)
    neg_semi = np.all(eigenvalues <= 0)

    if pos:
        return "Positive definite"
    if neg:
        return "Negative definite"
    if pos_semi:
        return "Positive semi-definite"
    if neg_semi:
        return "Negative semi-definite"
    
    # Check for mixture of signs
    has_pos = np.any(eigenvalues > 0)
    has_neg = np.any(eigenvalues < 0)
    if has_pos and has_neg:
        return "Indefinite"

    return None
