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

    # Define bins exactly every 10 units
    bins = np.arange(0, 110, 10)

    # Plot the histogram with black edge color
    plt.hist(student_grades, bins=bins, edgecolor='black')

    # Add labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Set exact x and y axis limits to match the reference plot
    plt.xlim(0, 100)
    plt.xticks(bins)
    plt.ylim(0, 30)

    # Display the plot
    plt.show()
