#!/usr/bin/env python3
"""
Contains the function l2_reg_create_layer
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
    Returns: the output of the new layer
    """
    # Define the weight initializer (He et al. / Variance Scaling)
    init = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg',
        distribution='uniform'
    )

    # Define the L2 regularizer
    reg = tf.keras.regularizers.L2(l2=lambtha)

    # Create the Dense layer
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=reg
    )

    return layer(prev)
