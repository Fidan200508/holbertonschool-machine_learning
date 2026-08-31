#!/usr/bin/env python3
"""Calculate posterior probabilities."""

intersection = __import__('1-intersection').intersection
marginal = __import__('2-marginal').marginal


def posterior(x, n, P, Pr):
    """Calculate posterior probabilities for each probability in P."""
    inter = intersection(x, n, P, Pr)
    marg = marginal(x, n, P, Pr)

    return inter / marg
