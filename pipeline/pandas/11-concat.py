#!/usr/bin/env python3
"""
Concatenates two DataFrames indexed by Timestamp with source labels.
"""

import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """
    Indexes both DataFrames on Timestamp, selects rows from df2 up to and
    including timestamp 1417411920, and concatenates df2 above df1 with
    source keys.

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame.
        df2 (pd.DataFrame): Bitstamp DataFrame.

    Returns:
        pd.DataFrame: Concatenated DataFrame with labeled keys.
    """
    df1 = index(df1)
    df2 = index(df2)

    df2 = df2[df2.index <= 1417411920]

    return pd.concat(
        [df2, df1],
        keys=["bitstamp", "coinbase"]
    )
