#!/usr/bin/env python3
"""Builds a neural network using the Keras Functional API."""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build and return a Keras neural network model."""
    inputs = K.Input(shape=(nx,))
    output = inputs

    for i in range(len(layers)):
        output = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(output)

        if i < len(layers) - 1:
            output = K.layers.Dropout(
                1 - keep_prob
            )(output)

    model = K.Model(
        inputs=inputs,
        outputs=output
    )

    return model
