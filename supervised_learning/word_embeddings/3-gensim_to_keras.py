#!/usr/bin/env python3
"""Convert a Gensim Word2Vec model to a Keras Embedding layer."""

import tensorflow as tf


def gensim_to_keras(model):
    """Convert a trained Gensim Word2Vec model to Keras Embedding.

    Args:
        model: Trained Gensim Word2Vec model.

    Returns:
        A trainable Keras Embedding layer.
    """
    weights = model.wv.vectors

    embedding = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )

    return embedding
