# Bitcoin Time Series Forecasting

This project uses an RNN architecture to forecast the closing price of Bitcoin for the following hour.

## Objective

The model uses the previous 24 hours of Bitcoin data to predict the close price of the following hour.

## Preprocessing

The raw Coinbase and Bitstamp datasets contain one observation per minute.

The preprocessing pipeline:

- removes missing values
- converts Unix timestamps to datetime
- sorts data chronologically
- resamples minute-level data into hourly data
- combines Coinbase and Bitstamp data
- uses Open, High, Low, Close, BTC volume, Currency volume, and Weighted Price
- normalizes features using statistics from the training data
- creates sequences containing the previous 24 hours
- uses the next-hour Close price as the target
- performs a chronological 80/20 train-validation split

The processed data is saved in btc_preprocessed.npz.

## Model

The forecasting model uses:

- LSTM layer with 64 units
- Dropout layer
- LSTM layer with 32 units
- Dense layer with 16 units
- Dense regression output with 1 unit

The model uses Mean Squared Error (MSE) as its loss function.

Training and validation data are provided using tf.data.Dataset.

## Files

- preprocess_data.py - preprocesses the raw Bitcoin data
- forecast_btc.py - creates, trains, and validates the model
- README.md - project documentation

## Usage

Preprocess the datasets:

    ./preprocess_data.py coinbaseUSD_1-min_data.csv bitstampUSD_1-min_data.csv

Then train the model:

    ./forecast_btc.py
