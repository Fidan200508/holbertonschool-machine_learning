#!/usr/bin/env python3
"""Convert Gensim Word2Vec embeddings to Keras."""

import tensorflow as tf


def gensim_to_keras(model):
    """Convert Gensim embeddings to a trainable Keras Embedding."""
    weights = model.wv.vectors

    return tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )
