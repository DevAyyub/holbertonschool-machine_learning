#!/usr/bin/env python3
"""
This module performs PCA on the Iris dataset and visualizes it in 3D.
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# Initialize the figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract x, y, and z coordinates from the pca_data columns
x = pca_data[:, 0]
y = pca_data[:, 1]
z = pca_data[:, 2]

# Plot the 3D scatter graph using the labels for color and plasma colormap
ax.scatter(x, y, z, c=labels, cmap='plasma')

# Add labels and title
ax.set_title('PCA of Iris Dataset')
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')

# Display the plot
plt.show()
