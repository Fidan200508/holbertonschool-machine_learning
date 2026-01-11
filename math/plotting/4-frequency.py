#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def frequency():
    """Plots a histogram of student scores for a project"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Define bins every 10 units from 0 to 100
    bins = range(0, 101, 10)

    # Plot the histogram
    # edgecolor='black' outlines the bars
    plt.hist(student_grades, bins=bins, edgecolor='black')

    # Set the labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Set x-axis ticks to match the bins for clarity
    plt.xticks(bins)

    plt.show()
