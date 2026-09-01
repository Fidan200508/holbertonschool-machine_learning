#!/usr/bin/env python3
"""Functions for saving and loading Keras model weights."""


def save_weights(network, filename, save_format='keras'):
    """Save the weights of a Keras model."""
    network.save_weights(
        filename,
        save_format=save_format
    )


def load_weights(network, filename):
    """Load weights into a Keras model."""
    network.load_weights(filename)
