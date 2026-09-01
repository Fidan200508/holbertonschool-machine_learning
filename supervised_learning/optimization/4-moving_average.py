#!/usr/bin/env python3
"""Calculates the weighted moving average of a data set."""


def moving_average(data, beta):
    """Calculate bias-corrected weighted moving averages."""
    moving_averages = []
    v = 0

    for i, value in enumerate(data, start=1):
        v = beta * v + (1 - beta) * value

        corrected = v / (1 - beta ** i)

        moving_averages.append(corrected)

    return moving_averages
