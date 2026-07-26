#!/usr/bin/env python3
"""Semantic search using sentence embeddings"""

import os
import tensorflow as tf
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """Performs semantic search on a corpus of documents"""

    # Load Universal Sentence Encoder
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    documents = []
    file_paths = []

    # Read all files from corpus directory
    for filename in os.listdir(corpus_path):
        path = os.path.join(corpus_path, filename)
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                documents.append(text)
                file_paths.append(path)

    if not documents:
        return None

    # Embed sentence and documents
    doc_embeddings = embed(documents)
    query_embedding = embed([sentence])

    # Compute cosine similarity
    similarities = tf.keras.losses.cosine_similarity(
        query_embedding, doc_embeddings
    )

    # cosine_similarity returns negative values → take argmin
    best_index = tf.argmin(similarities).numpy()

    return documents[best_index]
