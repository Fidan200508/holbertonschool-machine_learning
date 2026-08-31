#!/usr/bin/env python3
"""Preprocess Bitcoin time series data for forecasting."""

import glob
import sys

import numpy as np
import pandas as pd


FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume_(BTC)",
    "Volume_(Currency)",
    "Weighted_Price",
]

WINDOW = 24
TRAIN_RATIO = 0.8


def load_exchange(path):
    """Load and convert one exchange dataset to hourly data."""
    data = pd.read_csv(path)

    required = ["Timestamp"] + FEATURES
    data = data[required].copy()

    data = data.dropna()

    data["Timestamp"] = pd.to_datetime(
        data["Timestamp"],
        unit="s"
    )

    data = data.set_index("Timestamp")
    data = data.sort_index()

    hourly = data.resample("1h").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume_(BTC)": "sum",
            "Volume_(Currency)": "sum",
            "Weighted_Price": "mean",
        }
    )

    hourly = hourly.dropna()

    return hourly


def combine_exchanges(paths):
    """Load and combine hourly data from multiple exchanges."""
    frames = []

    for path in paths:
        print("Loading:", path)
        frame = load_exchange(path)
        frames.append(frame)

    combined = pd.concat(frames)

    combined = combined.groupby(
        combined.index
    ).agg(
        {
            "Open": "mean",
            "High": "max",
            "Low": "min",
            "Close": "mean",
            "Volume_(BTC)": "sum",
            "Volume_(Currency)": "sum",
            "Weighted_Price": "mean",
        }
    )

    combined = combined.sort_index()
    combined = combined.dropna()

    return combined


def make_sequences(data, split_index):
    """Create 24-hour input sequences and next-hour targets."""
    x_train = []
    y_train = []
    x_val = []
    y_val = []

    close_index = FEATURES.index("Close")

    for target_index in range(WINDOW, len(data)):
        sequence = data[
            target_index - WINDOW:target_index
        ]

        target = data[
            target_index,
            close_index
        ]

        if target_index < split_index:
            x_train.append(sequence)
            y_train.append(target)
        else:
            x_val.append(sequence)
            y_val.append(target)

    x_train = np.asarray(
        x_train,
        dtype=np.float32
    )

    y_train = np.asarray(
        y_train,
        dtype=np.float32
    )

    x_val = np.asarray(
        x_val,
        dtype=np.float32
    )

    y_val = np.asarray(
        y_val,
        dtype=np.float32
    )

    return x_train, y_train, x_val, y_val


def main():
    """Preprocess Bitcoin datasets and save the result."""
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = sorted(
            set(
                glob.glob("*coinbase*.csv")
                + glob.glob("*Coinbase*.csv")
                + glob.glob("*bitstamp*.csv")
                + glob.glob("*Bitstamp*.csv")
            )
        )

    if not paths:
        raise FileNotFoundError(
            "No Coinbase or Bitstamp CSV files were found."
        )

    dataframe = combine_exchanges(paths)

    raw_data = dataframe[
        FEATURES
    ].to_numpy(
        dtype=np.float64
    )

    split_index = int(
        len(raw_data) * TRAIN_RATIO
    )

    train_raw = raw_data[
        :split_index
    ]

    mean = train_raw.mean(
        axis=0
    )

    std = train_raw.std(
        axis=0
    )

    std[std == 0] = 1

    scaled_data = (
        raw_data - mean
    ) / std

    x_train, y_train, x_val, y_val = make_sequences(
        scaled_data,
        split_index
    )

    np.savez_compressed(
        "btc_preprocessed.npz",
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        mean=mean,
        std=std,
        features=np.asarray(FEATURES),
    )

    print()
    print("Preprocessing complete")
    print("----------------------")
    print("Hourly rows:", len(raw_data))
    print("X_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", x_val.shape)
    print("y_val:", y_val.shape)
    print("Saved: btc_preprocessed.npz")


if __name__ == "__main__":
    main()
