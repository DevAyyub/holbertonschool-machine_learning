#!/usr/bin/env python3

import numpy as np

oh_encode = __import__('24-one_hot_encode').one_hot_encode

# Using mock data similar to the example
Y = np.array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4])

print("Original labels:")
print(Y)
Y_one_hot = oh_encode(Y, 10)
print("\nOne-hot encoded matrix:")
print(Y_one_hot)
