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
        self.load_model()
        # Generate and set features
        self.generate_features()

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
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')

        x = vgg.input
        style_outputs = {name: None for name in self.style_layers}
        content_output = None

        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )(x)
            else:
                x = layer(x)

            if layer.name in self.style_layers:
                style_outputs[layer.name] = x
            if layer.name == self.content_layer:
                content_output = x

        model_outputs = [style_outputs[n] for n in self.style_layers]
        model_outputs.append(content_output)

        self.model = tf.keras.Model(inputs=vgg.input, outputs=model_outputs)
        self.model.trainable = False

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of a layer output
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
           len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        gram = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return gram / num_locations

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost
        Sets:
            gram_style_features: list of gram matrices for style image
            content_feature: content layer output for content image
        """
        # Get outputs from the model for the style image
        style_outputs = self.model(self.style_image)
        # Style outputs are the first len(style_layers) elements
        self.gram_style_features = [
            self.gram_matrix(out) for out in style_outputs[:-1]
        ]

        # Get outputs from the model for the content image
        content_outputs = self.model(self.content_image)
        # Content output is the last element
        self.content_feature = content_outputs[-1]
