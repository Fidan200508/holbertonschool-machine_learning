#!/usr/bin/env python3
"""Tests a Keras neural network model."""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Test a neural network and return its loss and accuracy."""
    return network.evaluate(
        data,
        labels,
        verbose=verbose
    )
