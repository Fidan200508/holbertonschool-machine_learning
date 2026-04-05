#!/usr/bin/env python3
"""
Convolutional layer backward propagation
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer

    Parameters:
    - dZ: numpy.ndarray of shape (m, h_new, w_new, c_new)
          Partial derivatives wrt the output of the conv layer
    - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
              Output of the previous layer
    - W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
         Convolution kernels
    - b: numpy.ndarray of shape (1, 1, 1, c_new)
         Biases applied to the convolution
    - padding: "same" or "valid"
    - stride: tuple of (sh, sw)

    Returns:
    - dA_prev: partial derivatives wrt previous layer
    - dW: partial derivatives wrt kernels
    - db: partial derivatives wrt biases
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == "same":
        pad_h = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pad_w = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        pad_h, pad_w = 0, 0

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant',
        constant_values=0
    )
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):  # loop over examples
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        if pad_h == 0 and pad_w == 0:
            dA_prev = dA_prev_pad
        else:
            dA_prev = dA_prev_pad[:, pad_h:-pad_h, pad_w:-pad_w, :]

    return dA_prev, dW, db
