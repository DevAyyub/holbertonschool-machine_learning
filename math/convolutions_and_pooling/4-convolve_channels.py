#!/usr/bin/env python3
"""
Module to perform a convolution on images with channels
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels
    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernel: numpy.ndarray with shape (kh, kw, c)
        padding: tuple (ph, pw), 'same', or 'valid'
        stride: tuple (sh, sw)
    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = (((h - 1) * sh + kh - h) // 2) + 1
        pw = (((w - 1) * sw + kw - w) // 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)

    out_h = ((h + 2 * ph - kh) // sh) + 1
    out_w = ((w + 2 * pw - kw) // sw) + 1

    convolved = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            v_start = i * sh
            h_start = j * sw
            image_slice = images_padded[:, v_start:v_start+kh,
                                        h_start:h_start+kw, :]
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2, 3))

    return convolved
