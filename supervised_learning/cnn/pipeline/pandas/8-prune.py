#!/usr/bin/env python3
"""
Removes rows with NaN values in the Close column.
"""


def prune(df):
    """
    Removes any entries where the Close column has NaN values.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame without NaN values in Close.
    """
    return df.dropna(subset=["Close"])
