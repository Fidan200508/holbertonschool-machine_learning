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

    df["Close"] = df["Close"].fillna(method="ffill")

    df[["High", "Low", "Open"]] = df[["High", "Low", "Open"]].fillna(
        df["Close"], axis=0
    )

    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df
