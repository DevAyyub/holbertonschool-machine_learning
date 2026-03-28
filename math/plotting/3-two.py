#!/usr/bin/env python3
"""
This module provides a function to plot two line graphs
representing the exponential decay of C-14 and Ra-226.
"""
import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    Plots the exponential decay of two radioactive elements
    with specific line styles, limits, and a legend.
    """
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plot y1 (dashed red) and y2 (solid green) with labels for the legend
    plt.plot(x, y1, 'r--', label='C-14')
    plt.plot(x, y2, 'g-', label='Ra-226')

    # Add labels and title
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of Radioactive Elements')

    # Limit the x and y axes
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Place the legend in the upper right hand corner
    plt.legend(loc='upper right')

    # Display the plot
    plt.show()
