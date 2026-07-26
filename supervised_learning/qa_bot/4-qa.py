#!/usr/bin/env python3
"""Multi-reference QA system"""

import os
import tensorflow as tf
import tensorflow_hub as hub

# import əvvəlki QA funksiyası
qa = __import__('0-qa').question_answer


def semantic_search(corpus_path, sentence):
    """Find most relevant document"""

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    documents = []

    for filename in os.listdir(corpus_path):
        path = os.path.join(corpus_path, filename)
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                documents.append(f.read())

    if not documents:
        return None

    doc_embeddings = embed(documents)
    query_embedding = embed([sentence])

    similarities = tf.keras.losses.cosine_similarity(
        query_embedding, doc_embeddings
    )

    best_index = tf.argmin(similarities).numpy()

    return documents[best_index]


def question_answer(corpus_path):
    """Answer questions using multiple documents"""

    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        question = input("Q: ")

        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        # 1. ən uyğun document tap
        reference = semantic_search(corpus_path, question)

        if reference is None:
            print("A: Sorry, I do not understand your question.")
            continue

        # 2. həmin documentdən cavab çıxar
        answer = qa(question, reference)

        if answer is None or answer.strip() == "":
            print("A: Sorry, I do not understand your question.")
        else:
            print("A:", answer)
