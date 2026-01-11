#!/usr/bin/env python3
"""
Sorts a DataFrame in reverse chronological order and transposes it.
"""


def flip_switch(df):
    """
    Sorts the DataFrame in reverse chronological order
    and returns its transpose.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    df = df.sort_index(ascending=False)
    return df.transpose()
