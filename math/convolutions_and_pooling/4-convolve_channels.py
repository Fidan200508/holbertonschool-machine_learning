#!/usr/bin/env python3
"""
Module to perform a convolution on images with channels.
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernel: numpy.ndarray with shape (kh, kw, c)
        padding: tuple (ph, pw), 'same', or 'valid'
        stride: tuple (sh, sw)

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    else:
        ph, pw = padding

    # Pad only height and width axes (1 and 2).
    # m (0) and c (3) axes receive no padding.
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)

    # Calculate output dimensions
    out_h = (h + (2 * ph) - kh) // sh + 1
    out_w = (w + (2 * pw) - kw) // sw + 1

    # Initialize output array
    output = np.zeros((m, out_h, out_w))

    # Perform convolution using 2 loops over the output grid
    for i in range(out_h):
        for j in range(out_w):
            h_s, w_s = i * sh, j * sw
            # Extract window: shape (m, kh, kw, c)
            window = images_padded[:, h_s:h_s + kh, w_s:w_s + kw, :]
            # Multiply by kernel and sum across axes 1, 2, and 3 (h, w, c)
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2, 3))

    return output
