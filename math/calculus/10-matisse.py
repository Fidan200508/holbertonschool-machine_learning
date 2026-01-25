#!/usr/bin/env python3
"""Module that computes the derivative of a polynomial"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    derivative = []
    for i in range(1, len(poly)):
        derivative.append(poly[i] * i)

    if all(coef == 0 for coef in derivative):
        return [0]

    return derivative
