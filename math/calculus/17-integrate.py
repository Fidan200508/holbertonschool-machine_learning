#!/usr/bin/env python3
"""Module that computes the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if not isinstance(poly, list) or not isinstance(C, (int, float)):
        return None

    if len(poly) == 0:
        return None

    integral = [C]

    for i, coef in enumerate(poly):
        value = coef / (i + 1)
        if value.is_integer():
            value = int(value)
        integral.append(value)

    # Remove trailing zeros to keep list as small as possible
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
