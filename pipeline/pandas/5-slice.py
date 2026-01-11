#!/usr/bin/env python3
"""
Slices a DataFrame to extract specific columns at regular intervals.
"""


def slice(df):
    """
    Extracts the High, Low, Close, and Volume_(BTC) columns
    and selects every 60th row.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Sliced DataFrame.
    """
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
