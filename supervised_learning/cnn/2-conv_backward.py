#!/usr/bin/env python3
"""
Module to perform back propagation over a convolutional layer
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    A_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                      mode='constant')
    dA_padded = np.pad(dA_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                       mode='constant')

    for i in range(h_new):
        for j in range(w_new):
            v_start = i * sh
            v_end = v_start + kh
            h_start = j * sw
            h_end = h_start + kw

            a_slice = A_padded[:, v_start:v_end, h_start:h_end, :]

            dW += np.sum(a_slice[:, :, :, :, np.newaxis] *
                         dZ[:, i:i+1, j:j+1, np.newaxis, :], axis=0)

            dA_padded[:, v_start:v_end, h_start:h_end, :] += np.sum(
                W * dZ[:, i:i+1, j:j+1, np.newaxis, :], axis=4)

    if padding == "same":
        dA_prev = dA_padded[:, ph:-ph if ph > 0 else None,
                            pw:-pw if pw > 0 else None, :]
    else:
        dA_prev = dA_padded

    return dA_prev, dW, db
