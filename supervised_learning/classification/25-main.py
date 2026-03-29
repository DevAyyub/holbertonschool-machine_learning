#!/usr/bin/env python3
import numpy as np

oh_encode = __import__('24-one_hot_encode').one_hot_encode
oh_decode = __import__('25-one_hot_decode').one_hot_decode

# Mock labels
Y = np.array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4])
print("Original labels:", Y)

# Encode then Decode
Y_one_hot = oh_encode(Y, 10)
Y_decoded = oh_decode(Y_one_hot)

print("Decoded labels: ", Y_decoded)

# Check if they match
if np.array_equal(Y, Y_decoded):
    print("Success: Decoding matches original!")
else:
    print("Failure: Decoded labels do not match.")
