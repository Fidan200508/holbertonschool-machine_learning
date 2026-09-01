#!/usr/bin/env python3
"""Creates mini-batches for mini-batch gradient descent."""

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Create shuffled mini-batches from X and Y."""
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    mini_batches = []
    m = X.shape[0]

    for start in range(0, m, batch_size):
        end = start + batch_size

        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]

        mini_batches.append(
            (X_batch, Y_batch)
        )

    return mini_batches
