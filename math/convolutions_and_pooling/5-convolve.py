#!/usr/bin/env python3
"""
Module to perform a convolution on images using multiple kernels.
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with multiple kernels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernels: numpy.ndarray with shape (kh, kw, c, nc)
        padding: tuple (ph, pw), 'same', or 'valid'
        stride: tuple (sh, sw)

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        # Standard calculation for 'same' with strides
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    else:
        ph, pw = padding

    # Apply zero padding to spatial dimensions
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)

    # Calculate output dimensions
    out_h = (h + (2 * ph) - kh) // sh + 1
    out_w = (w + (2 * pw) - kw) // sw + 1

    # Initialize output array with shape (m, out_h, out_w, nc)
    output = np.zeros((m, out_h, out_w, nc))

    # Perform convolution using 3 loops: height, width, and kernels
    for k in range(nc):
        current_kernel = kernels[:, :, :, k]
        for i in range(out_h):
            for j in range(out_w):
                h_s, w_s = i * sh, j * sw
                # Slice the window for all images: (m, kh, kw, c)
                window = images_padded[:, h_s:h_s + kh, w_s:w_s + kw, :]
                # Sum across height, width, and input channels
                output[:, i, j, k] = np.sum(window * current_kernel,
                                            axis=(1, 2, 3))

    return output
