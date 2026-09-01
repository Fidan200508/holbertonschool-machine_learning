#!/usr/bin/env python3
"""Calculates learning rate decay using inverse time decay."""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Calculate the stepwise inverse time learning rate decay."""
    step = np.floor(global_step / decay_step)

    return alpha / (1 + decay_rate * step)
