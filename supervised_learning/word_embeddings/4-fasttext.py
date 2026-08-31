#!/usr/bin/env python3
"""FastText training module."""

from gensim.models import FastText


def fasttext_model(
    sentences,
    vector_size=100,
    min_count=5,
    negative=5,
    window=5,
    cbow=True,
    epochs=5,
    seed=0,
    workers=1
):
    """Create, build, and train a FastText model.

    Args:
        sentences: Sentences to train on.
        vector_size: Dimensionality of the embeddings.
        min_count: Minimum frequency required for a word.
        negative: Number of negative samples.
        window: Maximum context window size.
        cbow: If True use CBOW, otherwise Skip-gram.
        epochs: Number of training epochs.
        seed: Random seed.
        workers: Number of worker threads.

    Returns:
        The trained FastText model.
    """
    model = FastText(
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
