#!/usr/bin/env python3
"""
5-all_in_one.py
Plots all previous graphs in a single figure.
"""

import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    Displays all five plots together in a 3x2 grid layout.
    """
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    fig = plt.figure(figsize=(6.4, 9.6))
    fig.suptitle("All in One", fontsize="x-small")

    # 1. Line graph
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax1.plot(np.arange(0, 11), y0, 'r-')
    ax1.set_title("Line Graph", fontsize="x-small")

    # 2. Scatter plot
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    ax2.scatter(x1, y1, c='m')
    ax2.set_title("Scatter Plot", fontsize="x-small")

    # 3. Change of scale
    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax3.plot(x2, y2)
    ax3.set_yscale("log")
    ax3.set_title("Change of Scale", fontsize="x-small")

    # 4. Two plots
    ax4 = plt.subplot2grid((3, 2), (1, 1))
    ax4.plot(x3, y31, 'r--', label='C-14')
    ax4.plot(x3, y32, 'g-', label='Ra-226')
    ax4.legend(fontsize="x-small")
    ax4.set_title("Two is Better Than One", fontsize="x-small")

    # 5. Histogram (spans two columns)
    ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    bins = np.arange(0, 101, 10)
    ax5.hist(student_grades, bins=bins, edgecolor='black')
    ax5.set_title("Frequency", fontsize="x-small")

    for ax in fig.axes:
        ax.set_xlabel(ax.get_xlabel(), fontsize="x-small")
        ax.set_ylabel(ax.get_ylabel(), fontsize="x-small")

    plt.tight_layout()
    plt.show()
