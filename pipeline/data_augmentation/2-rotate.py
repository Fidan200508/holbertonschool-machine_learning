#!/usr/bin/env python3
"""Rotate image module"""
import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise
    Args:
        image: a 3D tf.Tensor containing the image to rotate
    Returns:
        The rotated image
    """
    return tf.image.rot90(image, k=1)
