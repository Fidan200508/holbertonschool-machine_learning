#!/usr/bin/env python3
"""Expectation Maximization algorithm for GMM"""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs expectation maximization for a GMM

    Returns:
        pi, m, S, g, l or None, None, None, None, None on failure
    """

    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0 or
            not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)

    if pi is None:
        return None, None, None, None, None

    g, l = expectation(X, pi, m, S)

    if g is None:
        return None, None, None, None, None

    if verbose:
        print("Log Likelihood after 0 iterations: {:.5f}".format(l))

    for i in range(1, iterations + 1):
        prev_l = l

        pi, m, S = maximization(X, g)

        if pi is None:
            return None, None, None, None, None

        g, l = expectation(X, pi, m, S)

        if g is None:
            return None, None, None, None, None

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {:.5f}".format(i, l))

        if abs(l - prev_l) <= tol:
            if verbose and i % 10 != 0:
                print("Log Likelihood after {} iterations: {:.5f}".format(i, l))
            return pi, m, S, g, l

    if verbose and iterations % 10 != 0:
        print("Log Likelihood after {} iterations: {:.5f}".format(iterations, l))

    return pi, m, S, g, l
