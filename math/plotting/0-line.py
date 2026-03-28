#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def line():
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    # Plot y as a solid red line
    plt.plot(y, 'r-')
    
    # Limit the x-axis from 0 to 10
    plt.xlim(0, 10)
    
    # Display the plot
    plt.show()
