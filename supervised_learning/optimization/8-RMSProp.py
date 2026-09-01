#!/usr/bin/env python3
"""Creates a TensorFlow RMSProp optimizer."""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Set up the RMSProp optimization algorithm."""
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )

    return optimizer
