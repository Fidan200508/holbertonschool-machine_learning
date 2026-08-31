#!/usr/bin/env python3
"""Word2Vec training module."""

from gensim.models import Word2Vec


def word2vec_model(
    sentences,
    vector_size=100,
    min_count=5,
    window=5,
    negative=5,
    cbow=True,
    epochs=5,
    seed=0,
    workers=1
):
    """Create, build, and train a Word2Vec model.

    Args:
        sentences: Sentences to train on.
        vector_size: Dimensionality of word embeddings.
        min_count: Minimum frequency required for a word.
        window: Maximum context window size.
        negative: Number of negative samples.
        cbow: If True use CBOW, otherwise Skip-gram.
        epochs: Number of training epochs.
        seed: Random seed.
        workers: Number of worker threads.

    Returns:
        The trained Word2Vec model.
    """
    model = Word2Vec(
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
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
