#!/usr/bin/env python3
"""A module that converts Gensim embeddings to Keras."""

import tensorflow as tf


def gensim_to_keras(model):
    """Convert a Gensim Word2Vec model to a trainable Keras layer."""
    weights = model.wv.vectors

    embedding = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )

    return embedding
