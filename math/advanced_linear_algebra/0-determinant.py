#!/usr/bin/env python3
"""
Contains a function that calculates the determinant of a matrix
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix

    Args:
        matrix: list of lists whose determinant should be calculated

    Returns:
        The determinant of matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        if matrix == []:
            raise TypeError("matrix must be a list of lists")
        raise TypeError("matrix must be a list of lists")

    # Check if all elements are lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Handling the [[]] 0x0 case
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    n = len(matrix)

    # Check if square
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case for 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix
    if n == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][2 - 1])

    # Recursive step for nxn matrix using cofactor expansion
    det = 0
    for j in range(n):
        # Create a sub-matrix by removing the 0th row and jth column
        sub_matrix = [row[:j] + row[j + 1:] for row in matrix[1:]]
        # Alternate signs for expansion: (-1)^j * element * determinant of sub
        det += ((-1) ** j) * matrix[0][j] * determinant(sub_matrix)

    return det
