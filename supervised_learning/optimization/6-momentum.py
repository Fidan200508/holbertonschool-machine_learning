#!/usr/bin/env python3
"""Creates a TensorFlow momentum optimizer."""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Set up gradient descent with momentum optimization."""
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1
    )

    return optimizer
