#!/usr/bin/env python3
"""Random contrast module"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image
    Args:
        image: a 3D tf.Tensor containing the image to adjust
        lower: float, lower bound of the random contrast factor
        upper: float, upper bound of the random contrast factor
    Returns:
        The contrast-adjusted image
    """
    return tf.image.random_contrast(image, lower, upper)
