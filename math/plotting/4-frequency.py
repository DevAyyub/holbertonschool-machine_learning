#!/usr/bin/env python3
"""
This module provides a function to plot a histogram of student grades.
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student grades with bins every 10 units
    and a black outline on the bars.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Define the bins every 10 units from 0 to 100
    bins = range(0, 110, 10)

    # Plot the histogram with black edge color
    plt.hist(student_grades, bins=bins, edgecolor='black')

    # Add labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Set x-axis limits and tick marks to match the bins
    plt.xlim(0, 100)
    plt.xticks(bins)

    # Display the plot
    plt.show()
