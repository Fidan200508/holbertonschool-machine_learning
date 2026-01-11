#!/usr/bin/env python3
"""
0-line.py
Plots a line graph of y = x^3 from 0 to 10.
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plots y as a solid red line graph.
    The x-axis ranges from 0 to 10.
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 11)
    plt.plot(x, y, 'r-')
    plt.xlim(0, 10)

    plt.show()
