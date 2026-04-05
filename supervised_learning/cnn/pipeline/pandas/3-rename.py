#!/usr/bin/env python3
"""
Renames the Timestamp column and converts it to datetime.
"""

import pandas as pd


def rename(df):
    """
    Renames the Timestamp column to Datetime, converts it to datetime,
    and returns only the Datetime and Close columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing a Timestamp column.

    Returns:
        pd.DataFrame: Modified DataFrame with Datetime and Close columns.
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    return df[["Datetime", "Close"]]
