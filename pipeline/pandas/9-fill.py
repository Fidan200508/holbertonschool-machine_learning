#!/usr/bin/env python3
"""
Fills missing values in a DataFrame using specific rules.
"""


def fill(df):
    """
    Removes the Weighted_Price column, fills missing values in Close,
    High, Low, Open, and volume columns according to given rules.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = df.drop(columns=["Weighted_Price"])

    # Fill Close with previous row value
    df["Close"] = df["Close"].fillna(method="ffill")

    # Fill High, Low, Open with same-row Close value
    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])
    df["Open"] = df["Open"].fillna(df["Close"])

    # Fill volume columns with 0
    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df
