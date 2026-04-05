#!/usr/bin/env python3
"""
1-pool_forward.py
Performs forward propagation over a pooling layer in a CNN
"""

import numpy as np

def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer

    Parameters:
    - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
              containing the output of the previous layer
    - kernel_shape: tuple (kh, kw) size of the pooling kernel
    - stride: tuple (sh, sw) strides for height and width
    - mode: 'max' or 'avg' for max or average pooling

    Returns:
    - output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    h_new = int((h_prev - kh) / sh) + 1
    w_new = int((w_prev - kw) / sw) + 1

    # Initialize output
    A = np.zeros((m, h_new, w_new, c_prev))

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            slice_prev = A_prev[:, vert_start:vert_end, horiz_start:horiz_end, :]

            if mode == 'max':
                A[:, i, j, :] = np.max(slice_prev, axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(slice_prev, axis=(1, 2))

    return A
