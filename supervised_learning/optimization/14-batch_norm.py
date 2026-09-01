#!/usr/bin/env python3
"""Creates a neural network layer with batch normalization."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Create a dense layer followed by batch normalization."""
    initializer = tf.keras.initializers.VarianceScaling(
        mode='fan_avg'
    )

    dense = tf.keras.layers.Dense(
        n,
        kernel_initializer=initializer
    )

    Z = dense(prev)

    mean, variance = tf.nn.moments(
        Z,
        axes=[0]
    )

    gamma = tf.Variable(
        tf.ones([n]),
        trainable=True
    )

    beta = tf.Variable(
        tf.zeros([n]),
        trainable=True
    )

    Z_norm = tf.nn.batch_normalization(
        Z,
        mean,
        variance,
        beta,
        gamma,
        1e-7
    )

    return activation(Z_norm)
