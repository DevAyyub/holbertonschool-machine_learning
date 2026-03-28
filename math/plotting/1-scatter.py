#!/usr/bin/env python3
"""
This module provides a function to plot a scatter graph
representing height vs weight.
"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Plots a scatter graph of height versus weight using randomly
    generated multivariate normal distribution data.
    """
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    # Plot the data as magenta points
    plt.scatter(x, y, c='magenta')

    # Add labels and title
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.title("Men's Height vs Weight")

    # Display the plot
    plt.show()
