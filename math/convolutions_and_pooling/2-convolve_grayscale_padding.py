#!/usr/bin/env python3
"""
Module to perform a convolution on grayscale images with custom padding.
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)
        padding: tuple of (ph, pw)

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Apply the custom zero padding to axes 1 (height) and 2 (width)
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    # Calculate output dimensions based on the padded image
    out_h = h + (2 * ph) - kh + 1
    out_w = w + (2 * pw) - kw + 1

    # Initialize output array
    output = np.zeros((m, out_h, out_w))

    # Perform convolution using 2 loops over output height and width
    for i in range(out_h):
        for j in range(out_w):
            # Slice the window from the padded images
            window = images_padded[:, i:i+kh, j:j+kw]
            # Element-wise multiply and sum for all images simultaneously
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
