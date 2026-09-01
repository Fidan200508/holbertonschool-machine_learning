#!/usr/bin/env python3
"""Functions for saving and loading Keras model configurations."""

import tensorflow.keras as K


def save_config(network, filename):
    """Save a Keras model configuration in JSON format."""
    config = network.to_json()

    with open(filename, 'w') as file:
        file.write(config)


def load_config(filename):
    """Load a Keras model from a JSON configuration."""
    with open(filename, 'r') as file:
        config = file.read()

    return K.models.model_from_json(config)
