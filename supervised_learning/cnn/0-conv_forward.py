#!/usr/bin/env python3
"""
Module to perform forward propagation over a convolutional layer
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network
    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
        b: numpy.ndarray of shape (1, 1, 1, c_new)
        activation: function applied to the convolution
        padding: string, "same" or "valid"
        stride: tuple of (sh, sw)
    Returns:
        the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    # Calculate output dimensions
    h_out = int((h_prev + 2 * ph - kh) / sh) + 1
    w_out = int((w_prev + 2 * pw - kw) / sw) + 1

    # Apply padding
    A_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                      mode='constant', constant_values=0)

    # Initialize output array
    output = np.zeros((m, h_out, w_out, c_new))

    # Perform convolution using broadcasting over examples and output channels
    for i in range(h_out):
        for j in range(w_out):
            v_start = i * sh
            v_end = v_start + kh
            h_start = j * sw
            h_end = h_start + kw

            # slice has shape (m, kh, kw, c_prev)
            img_slice = A_padded[:, v_start:v_end, h_start:h_end, :]

            # Multiply slice by kernels (kh, kw, c_prev, c_new)
            # Sum over height, width, and input channel axes
            output[:, i, j, :] = np.sum(img_slice[:, :, :, :, np.newaxis] *
                                        W, axis=(1, 2, 3))

    return activation(output + b)
