#!/usr/bin/env python3
"""Module that contains function to perform element-wise operations"""


def np_elementwise(mat1, mat2):
    """Returns element-wise sum, difference, product and quotient"""
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
