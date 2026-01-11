#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file(
    'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv',
    ','
)

# Remove Weighted_Price
df = df.drop(columns=["Weighted_Price"])

# Rename Timestamp to Date
df = df.rename(columns={"Timestamp": "Date"})

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], unit="s")

# Set Date as index
df = df.set_index("Date")

# Fill missing Close values with previous row
df["Close"] = df["Close"].fillna(method="ffill")

# Fill High, Low, Open with same-row Close
df["High"] = df["High"].fillna(df["Close"])
df["Low"] = df["Low"].fillna(df["Close"])
df["Open"] = df["Open"].fillna(df["Close"])

# Fill volume columns with 0
df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

# Keep data from 2017 onwards
df = df[df.index.year >= 2017]

# Resample daily and aggregate
df = df.resample("D").agg({
    "High": "max",
    "Low": "min",
    "Open": "mean",
    "Close": "mean",
    "Volume_(BTC)": "sum",
    "Volume_(Currency)": "sum",
})

# Display transformed DataFrame
print(df)

# Plot
df.plot(figsize=(12, 6))
plt.show()
