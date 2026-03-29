#!/usr/bin/env python3
"""
Module to perform pooling on images
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images
    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernel_shape: tuple (kh, kw)
        stride: tuple (sh, sw)
        mode: 'max' or 'avg'
    Returns:
        numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = ((h - kh) // sh) + 1
    out_w = ((w - kw) // sw) + 1

    pooled = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            v_start = i * sh
            h_start = j * sw
            image_slice = images[:, v_start:v_start+kh, h_start:h_start+kw, :]

            if mode == 'max':
                pooled[:, i, j, :] = np.max(image_slice, axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(image_slice, axis=(1, 2))

    return pooled
