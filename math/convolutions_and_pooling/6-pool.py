#!/usr/bin/env python3
"""
Module to perform pooling on images.
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernel_shape: tuple of (kh, kw)
        stride: tuple of (sh, sw)
        mode: 'max' or 'avg'

    Returns:
        A numpy.ndarray containing the pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    # Initialize output array
    output = np.zeros((m, out_h, out_w, c))

    # Perform pooling using 2 loops over the output grid
    for i in range(out_h):
        for j in range(out_w):
            h_s, w_s = i * sh, j * sw
            # Extract the window for all images and all channels
            window = images[:, h_s:h_s + kh, w_s:w_s + kw, :]

            if mode == 'max':
                # Max over the spatial axes (1 and 2)
                output[:, i, j, :] = np.max(window, axis=(1, 2))
            elif mode == 'avg':
                # Mean over the spatial axes (1 and 2)
                output[:, i, j, :] = np.mean(window, axis=(1, 2))

    return output
