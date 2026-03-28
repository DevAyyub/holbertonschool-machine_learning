#!/usr/bin/env python3
"""
This module provides a function to plot a scatter plot of sampled
elevations on a mountain, utilizing a color gradient.
"""
import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """
    Plots a scatter plot of a mountain elevation gradient
    using a colorbar to display the elevation (z) values.
    """
    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    # Plot the scatter graph mapping z values to colors
    plt.scatter(x, y, c=z)

    # Add labels and title
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.title('Mountain Elevation')

    # Add a colorbar and label it
    plt.colorbar(label='elevation (m)')

    # Display the plot
    plt.show()
