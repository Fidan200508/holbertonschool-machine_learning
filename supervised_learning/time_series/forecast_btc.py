#!/usr/bin/env python3
"""Train an RNN model to forecast Bitcoin closing price."""

import numpy as np
import tensorflow as tf


BATCH_SIZE = 64
EPOCHS = 20
BUFFER_SIZE = 10000
WINDOW = 24


def create_datasets(data):
    """Create TensorFlow datasets for training and validation."""
    x_train = data["X_train"]
    y_train = data["y_train"]

    x_val = data["X_val"]
    y_val = data["y_val"]

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    )

    train_dataset = train_dataset.shuffle(
        min(
            BUFFER_SIZE,
            len(x_train)
        )
    )

    train_dataset = train_dataset.batch(
        BATCH_SIZE
    )

    train_dataset = train_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (x_val, y_val)
    )

    validation_dataset = validation_dataset.batch(
        BATCH_SIZE
    )

    validation_dataset = validation_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return train_dataset, validation_dataset


def build_model(number_features):
    """Build the LSTM Bitcoin forecasting model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(
                    WINDOW,
                    number_features
                )
            ),

            tf.keras.layers.LSTM(
                64,
                return_sequences=True
            ),

            tf.keras.layers.Dropout(
                0.2
            ),

            tf.keras.layers.LSTM(
                32
            ),

            tf.keras.layers.Dense(
                16,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                1
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    return model


def main():
    """Train and validate the BTC forecasting model."""
    data = np.load(
        "btc_preprocessed.npz",
        allow_pickle=True
    )

    train_dataset, validation_dataset = create_datasets(
        data
    )

    number_features = data[
        "X_train"
    ].shape[-1]

    model = build_model(
        number_features
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            "btc_model.keras",
            monitor="val_loss",
            save_best_only=True
        ),
    ]

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    loss, mae = model.evaluate(
        validation_dataset
    )

    print()
    print("Validation results")
    print("------------------")
    print("Validation MSE:", loss)
    print("Validation MAE:", mae)

    if len(data["X_val"]) > 0:
        sample = data[
            "X_val"
        ][:1]

        prediction_scaled = model.predict(
            sample,
            verbose=0
        )[0, 0]

        close_index = 3

        mean = data[
            "mean"
        ]

        std = data[
            "std"
        ]

        prediction = (
            prediction_scaled
            * std[close_index]
            + mean[close_index]
        )

        actual = (
            data["y_val"][0]
            * std[close_index]
            + mean[close_index]
        )

        print()
        print(
            "Predicted next-hour close: "
            "${:.2f}".format(prediction)
        )

        print(
            "Actual next-hour close:    "
            "${:.2f}".format(actual)
        )


if __name__ == "__main__":
    main()
