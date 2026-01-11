#!/usr/bin/env python3
"""
Computes descriptive statistics for a DataFrame.
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns
    except the Timestamp column.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing descriptive statistics.
    """
    return df.drop(columns=["Timestamp"]).describe()
