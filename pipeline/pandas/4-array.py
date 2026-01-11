#!/usr/bin/env python3
"""
Selects the last 10 rows of High and Close columns as a NumPy array.
"""

import numpy as np


def array(df):
    """
    Selects the last 10 rows of the High and Close columns
    and returns them as a NumPy array.

    Args:
        df (pd.DataFrame): Input DataFrame containing High and Close columns.

    Returns:
        np.ndarray: NumPy array of the selected values.
    """
    return df[["High", "Close"]].tail(10).to_numpy()
