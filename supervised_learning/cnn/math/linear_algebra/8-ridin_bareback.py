#!/usr/bin/env python3
"""Module for multiplying two matrices"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication"""
    if len(mat1[0]) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):              # rows of mat1
        new_row = []
        for j in range(len(mat2[0])):       # columns of mat2
            total = 0
            for k in range(len(mat2)):      # columns of mat1 / rows of mat2
                total += mat1[i][k] * mat2[k][j]
            new_row.append(total)
        result.append(new_row)

    return result
