#!/usr/bin/env python3
"""
Module to perform back propagation over a pooling layer
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network
    """
    m, h_new, w_new, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            v_start = i * sh
            v_end = v_start + kh
            h_start = j * sw
            h_end = h_start + kw

            if mode == 'max':
                for ex in range(m):
                    for ch in range(c):
                        a_prev_slice = A_prev[ex, v_start:v_end,
                                              h_start:h_end, ch]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[ex, v_start:v_end, h_start:h_end, ch] += (
                            mask * dA[ex, i, j, ch]
                        )

            elif mode == 'avg':
                avg_grad = dA[:, i, j, :] / (kh * kw)
                dA_prev[:, v_start:v_end, h_start:h_end, :] += (
                    avg_grad[:, np.newaxis, np.newaxis, :]
                )

    return dA_prev
