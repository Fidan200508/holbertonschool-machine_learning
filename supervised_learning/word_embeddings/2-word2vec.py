#!/usr/bin/env python3
"""Word2Vec model creation module."""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create and build a Gensim Word2Vec model.

    Args:
        sentences: List of sentences to process.
        vector_size: Dimensionality of the embedding layer.
        min_count: Minimum number of occurrences of a word.
        window: Maximum distance between words.
        negative: Number of negative samples.
        cbow: True for CBOW and False for Skip-gram.
        epochs: Number of training epochs.
        seed: Seed for the random number generator.
        workers: Number of worker threads.

    Returns:
        The Word2Vec model.
    """
    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=not cbow,
        seed=seed,
        workers=workers,
        epochs=epochs,
        sorted_vocab=0
    )

    model.build_vocab(sentences)

    model.wv.sort_by_descending_frequency()

    return model
