#!/usr/bin/env python3
"""
Module to perform a same convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding needed to keep same dimensions
    # Using // 2 handles both even and odd kernel sizes correctly
    ph = kh // 2
    pw = kw // 2

    # Apply zero padding to the height and width axes (axes 1 and 2)
    # The padding for the 'm' axis is (0, 0)
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    # Initialize output with same h and w as input
    output = np.zeros((m, h, w))

    # Perform convolution using 2 loops over the output height and width
    for i in range(h):
        for j in range(w):
            # Extract window from padded images
            window = images_padded[:, i:i+kh, j:j+kw]
            # Element-wise multiply and sum across the spatial dimensions
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
