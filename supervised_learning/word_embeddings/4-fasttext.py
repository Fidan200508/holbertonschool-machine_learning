#!/usr/bin/env python3
"""FastText training module."""

import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create, build, and train a FastText model.

    Args:
        sentences: List of sentences to train on.
        vector_size: Dimensionality of the embedding layer.
        min_count: Minimum number of occurrences of a word.
        negative: Size of negative sampling.
        window: Maximum distance between words.
        cbow: True for CBOW and False for Skip-gram.
        epochs: Number of training epochs.
        seed: Seed for the random number generator.
        workers: Number of worker threads.

    Returns:
        The trained FastText model.
    """
    model = gensim.models.FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=not cbow,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
