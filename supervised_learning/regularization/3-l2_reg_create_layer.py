#!/usr/bin/env python3
"""Create a TensorFlow layer with L2 regularization"""
import tensorflow as tf

def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a layer in TensorFlow with L2 regularization

    prev: tensor, output of the previous layer
    n: number of nodes in the new layer
    activation: activation function to use
    lambtha: L2 regularization parameter

    Returns: output tensor of the new layer
    """
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.L2(l2=lambtha)
    )(prev)
    return layer
