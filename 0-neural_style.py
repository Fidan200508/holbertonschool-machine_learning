#!/usr/bin/env python3
"""
Contains the NST class for Neural Style Transfer
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    NST class that performs tasks for neural style transfer
    """
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for NST

        Args:
            style_image: image used as style reference (numpy.ndarray)
            content_image: image used as content reference (numpy.ndarray)
            alpha: weight for content cost
            beta: weight for style cost
        """
        # Validation for style_image
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        # Validation for content_image
        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        # Validation for alpha
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        # Validation for beta
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Set attributes
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that pixel values are [0, 1]
        and the largest side is 512 pixels.

        Args:
            image: numpy.ndarray of shape (h, w, 3)

        Returns:
            scaled image as a tf.Tensor with shape (1, h_new, w_new, 3)
        """
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape

        # Calculate new dimensions
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        # Resize using bicubic interpolation
        # tf.image.resize expects a batch dimension or just (h, w)
        image_resized = tf.image.resize(
            image,
            size=(h_new, w_new),
            method=tf.image.ResizeMethod.BICUBIC
        )

        # Rescale pixel values from [0, 255] to [0, 1]
        image_resized = image_resized / 255.0

        # Clip values to ensure they stay within [0, 1] after interpolation
        image_resized = tf.clip_by_value(image_resized, 0, 1)

        # Add batch dimension: (1, h_new, w_new, 3)
        return tf.expand_dims(image_resized, axis=0)
