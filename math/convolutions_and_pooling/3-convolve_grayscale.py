#!/usr/bin/env python3
"""
Module to perform a strided convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with stride and padding.

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)
        padding: tuple (ph, pw), 'same', or 'valid'
        stride: tuple (sh, sw)

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    else:
        ph, pw = padding

    # Apply zero padding
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    # Calculate output dimensions
    out_h = (h + (2 * ph) - kh) // sh + 1
    out_w = (w + (2 * pw) - kw) // sw + 1

    # Initialize output array
    output = np.zeros((m, out_h, out_w))

    # Perform convolution using 2 loops over the output grid
    for i in range(out_h):
        for j in range(out_w):
            # Calculate the starting position based on stride
            h_start = i * sh
            w_start = j * sw
            # Slice the window from the padded images
            window = images_padded[:, h_start:h_start + kh, w_start:w_start + kw]
            # Multiply and sum
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
