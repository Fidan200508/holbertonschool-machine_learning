#!/usr/bin/env python3
"""
Sorts a DataFrame by the High column in descending order.
"""


def high(df):
    """
    Sorts the DataFrame by the High price in descending order.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame sorted by High price.
    """
    return df.sort_values(by="High", ascending=False)
