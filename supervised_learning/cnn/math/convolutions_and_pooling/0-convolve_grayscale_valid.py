#!/usr/bin/env python3
"""
Module to perform a valid convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate output dimensions
    out_h = h - kh + 1
    out_w = w - kw + 1

    # Initialize the output array with zeros
    output = np.zeros((m, out_h, out_w))

    # Perform convolution using 2 loops (height and width of output)
    for i in range(out_h):
        for j in range(out_w):
            # Extract the window from all images at once
            window = images[:, i:i+kh, j:j+kw]
            # Multiply by kernel and sum across the spatial axes (1 and 2)
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
