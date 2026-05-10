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
        """
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        # Load the model and save to instance attribute
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales image to [0, 1] with max side 512
        """
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        if h > w:
            h_new, w_new = 512, int(w * (512 / h))
        else:
            w_new, h_new = 512, int(h * (512 / w))

        image_resized = tf.image.resize(
            image, size=(h_new, w_new), method='bicubic'
        )
        image_resized = tf.clip_by_value(image_resized / 255.0, 0, 1)
        return tf.expand_dims(image_resized, axis=0)

    def load_model(self):
        """
        Creates the model used to calculate cost using VGG19
        """
        # Load pre-trained VGG19 without the top classification layers
        vgg = tf.keras.applications.vGG19(
            include_top=False,
            weights='imagenet'
        )

        # Replace MaxPool layers with AveragePool layers for smoother gradients
        # as suggested in the original Leon Gatys paper
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg.trainable = False

        # Get output tensors for style and content layers
        outputs = [vgg.get_layer(name).output for name in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)

        # Build the final functional model
        model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

        # Ensure the model is not trainable to save memory and compute
        for layer in model.layers:
            layer.trainable = False

        self.model = model
