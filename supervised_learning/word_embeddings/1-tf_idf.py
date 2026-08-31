#!/usr/bin/env python3
"""TF-IDF embedding module."""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Create a TF-IDF embedding matrix.

    Args:
        sentences: List of sentences to analyze.
        vocab: Vocabulary words to use. If None, all words are used.

    Returns:
        embeddings: TF-IDF embedding matrix.
        features: Features used in the embedding matrix.
    """
    vectorizer = TfidfVectorizer(
        vocabulary=vocab
    )

    embeddings = vectorizer.fit_transform(
        sentences
    ).toarray()

    try:
        features = vectorizer.get_feature_names_out()
    except AttributeError:
        features = vectorizer.get_feature_names()

    return embeddings, features
