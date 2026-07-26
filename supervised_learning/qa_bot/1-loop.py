#!/usr/bin/env python3
"""Simple QA loop"""

def main():
    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        question = input("Q: ")

        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        print("A:")


if __name__ == "__main__":
    main()
