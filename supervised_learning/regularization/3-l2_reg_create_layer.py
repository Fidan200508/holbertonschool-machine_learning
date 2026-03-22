#!/usr/bin/env python3
"""
Module to create a TensorFlow layer with L2 regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in TensorFlow that includes L2 regularization

    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function that should be used on the layer
        lambtha: L2 regularization parameter

    Returns:
        The output of the new layer
    }
    """
    # Initialize the L2 regularizer with the given lambtha
    regularizer = tf.keras.regularizers.L2(lambtha)

    # Initialize weights using He Normal (standard for deep networks)
    # for consistency with most Holberton/ALX projects
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg',
        distribution='uniform'
    )

    # Create the Dense layer
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=regularizer,
        kernel_initializer=initializer
    )

    return layer(prev)
