#!/usr/bin/env python3
"""
Contains the function dropout_create_layer
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout
    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function for the new layer
        keep_prob: probability that a node will be kept
        training: boolean indicating whether the model is in training mode
    Returns: the output of the new layer
    """
    # Initialize weights using VarianceScaling (He et al.)
    init = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg',
        distribution='uniform'
    )

    # Define the Dense layer
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )

    # Pass the previous layer output through the dense layer
    output = layer(prev)

    # Create the Dropout layer. Note: rate = 1 - keep_prob
    # The training parameter determines if dropout is actually applied
    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)

    return dropout(output, training=training)
