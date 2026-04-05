#!/usr/bin/env python3
import numpy as np

def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer.

    dA: np.ndarray of shape (m, h_new, w_new, c_new)
        Partial derivatives w.r.t the output of the pooling layer
    A_prev: np.ndarray of shape (m, h_prev, w_prev, c)
        Output of the previous layer
    kernel_shape: tuple (kh, kw)
        Size of the pooling kernel
    stride: tuple (sh, sw)
        Stride for pooling
    mode: 'max' or 'avg'
        Pooling mode

    Returns:
        dA_prev: np.ndarray of shape (m, h_prev, w_prev, c)
        Partial derivatives w.r.t the previous layer
    """
    m, h_new, w_new, c = dA.shape
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Initialize dA_prev with zeros
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    if mode == 'max':
                        # Create mask of max value
                        A_slice = A_prev[i, h_start:h_end, w_start:w_end, ch]
                        mask = (A_slice == np.max(A_slice))
                        dA_prev[i, h_start:h_end, w_start:w_end, ch] += mask * dA[i, h, w, ch]
                    elif mode == 'avg':
                        # Distribute gradient evenly
                        da = dA[i, h, w, ch] / (kh * kw)
                        dA_prev[i, h_start:h_end, w_start:w_end, ch] += np.ones((kh, kw)) * da

    return dA_prev
