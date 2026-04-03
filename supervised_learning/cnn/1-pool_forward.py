#!/usr/bin/env python3
"""
Module to perform forward propagation over a pooling layer
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_out = int((h_prev - kh) / sh) + 1
    w_out = int((w_prev - kw) / sw) + 1

    output = np.zeros((m, h_out, w_out, c_prev))

    for i in range(h_out):
        for j in range(w_out):
            v_start = i * sh
            v_end = v_start + kh
            h_start = j * sw
            h_end = h_start + kw

            img_slice = A_prev[:, v_start:v_end, h_start:h_end, :]

            if mode == 'max':
                output[:, i, j, :] = np.max(img_slice, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(img_slice, axis=(1, 2))

    return output
