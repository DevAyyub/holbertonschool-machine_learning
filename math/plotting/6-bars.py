#!/usr/bin/env python3
"""
This module provides a function to plot a stacked bar graph.
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plots a stacked bar graph representing the quantity of various
    fruits possessed by different people.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    # Initialize an array of zeros to track the bottom of each stacked bar
    bottom = np.zeros(len(people))

    # Loop through each row (fruit type) and plot it
    for i in range(len(fruit)):
        plt.bar(people, fruit[i], width=0.5, bottom=bottom,
                color=colors[i], label=fruit_names[i])
        # Update the bottom tracker for the next iteration
        bottom += fruit[i]

    # Add labels, title, legend, and axis limits/ticks
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.legend()

    # Display the plot
    plt.show()
