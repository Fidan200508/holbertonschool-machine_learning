#!/usr/bin/env python3
"""Makes predictions using a Keras neural network."""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Make predictions using a neural network."""
    return network.predict(
        data,
        verbose=verbose
    )
