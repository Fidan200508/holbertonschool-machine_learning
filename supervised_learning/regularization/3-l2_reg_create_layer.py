
 #!/usr/bin/env python3
"""Module to create a layer with L2 regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in TensorFlow that includes L2 regularization

    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function for the new layer
        lambtha: L2 regularization parameter

    Returns:
        the output of the new layer
    """
    # He et al. initialization is standard for layers with activations
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg'
    )

    # Define the L2 regularizer
    regularizer = tf.keras.regularizers.L2(l2=lambtha)

    # Create the Dense layer with kernel_regularizer
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )

    return layer(prev)
