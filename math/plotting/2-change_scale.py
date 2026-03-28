#!/usr/bin/env python3
"""
This module provides a function to plot an exponential decay
graph with a logarithmically scaled y-axis.
"""
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plots the exponential decay of C-14, setting the y-axis
    to a logarithmic scale and limiting the x-axis.
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plot x vs y as a line graph
    plt.plot(x, y)

    # Add labels and title
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of C-14')

    # Set the y-axis to logarithmic scale
    plt.yscale('log')

    # Limit the x-axis from 0 to 28650
    plt.xlim(0, 28650)

    # Display the plot
    plt.show()
