#!/usr/bin/env python3
"""Bag of words embedding module."""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """Create a bag of words embedding matrix.

    Args:
        sentences: List of sentences to analyze.
        vocab: Vocabulary words to use. If None, all words are used.

    Returns:
        embeddings: Matrix containing word counts for each sentence.
        features: Features used in the embedding matrix.
    """
    vectorizer = CountVectorizer(
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
