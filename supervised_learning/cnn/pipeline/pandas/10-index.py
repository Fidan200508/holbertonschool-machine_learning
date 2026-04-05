#!/usr/bin/env python3
"""
Sets the Timestamp column as the index of a DataFrame.
"""


def index(df):
    """
    Sets the Timestamp column as the index of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame indexed by Timestamp.
    """
    return df.set_index("Timestamp")
