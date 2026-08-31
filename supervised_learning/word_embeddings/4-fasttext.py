#!/usr/bin/env python3
"""FastText training module."""

import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create, build, and train a FastText model.

    Args:
        sentences: List of sentences to train on.
        vector_size: Dimensionality of the embedding layer.
        min_count: Minimum number of occurrences for a word.
        negative: Number of negative samples.
        window: Maximum distance between current and predicted word.
        cbow: True for CBOW and False for Skip-gram.
        epochs: Number of training epochs.
        seed: Random seed.
        workers: Number of worker threads.

    Returns:
        The trained FastText model.
    """
    model = gensim.models.FastText(
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=0 if cbow else 1,
        seed=seed,
        workers=workers
    )

    model.build_vocab(sentences)

    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
