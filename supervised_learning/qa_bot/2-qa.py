#!/usr/bin/env python3
"""Answer loop using QA model"""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """Answers questions from a reference text"""

    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        question = input("Q: ")

        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        answer = question_answer(question, reference)

        if answer is None or answer.strip() == "":
            print("A: Sorry, I do not understand your question.")
        else:
            print("A:", answer)
