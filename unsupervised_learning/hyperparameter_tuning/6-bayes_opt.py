#!/usr/bin/env python3
"""
Bayesian Optimization with GPyOpt
"""

import GPyOpt
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((-1, 784)) / 255.0
X_test = X_test.reshape((-1, 784)) / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


def objective(x):
    """
    x:
    learning_rate
    units1
    units2
    dropout
    batch_size
    l2_reg
    """

    lr = float(x[:, 0])
    units1 = int(x[:, 1])
    units2 = int(x[:, 2])
    dropout_rate = float(x[:, 3])
    batch_size = int(x[:, 4])
    reg = float(x[:, 5])

    model = Sequential()

    model.add(
        Dense(
            units1,
            activation="relu",
            kernel_regularizer=l2(reg),
            input_shape=(784,)
        )
    )

    model.add(Dropout(dropout_rate))

    model.add(
        Dense(
            units2,
            activation="relu",
            kernel_regularizer=l2(reg)
        )
    )

    model.add(Dropout(dropout_rate))

    model.add(Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    filename = (
        f"lr_{lr:.5f}_u1_{units1}"
        f"_u2_{units2}_dr_{dropout_rate:.2f}"
        f"_bs_{batch_size}_reg_{reg:.5f}.h5"
    )

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filename,
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks
    )

    best_acc = max(history.history["val_accuracy"])

    return -best_acc


domain = [
    {
        "name": "lr",
        "type": "continuous",
        "domain": (1e-4, 1e-2)
    },
    {
        "name": "units1",
        "type": "discrete",
        "domain": (32, 64, 128, 256)
    },
    {
        "name": "units2",
        "type": "discrete",
        "domain": (16, 32, 64, 128)
    },
    {
        "name": "dropout",
        "type": "continuous",
        "domain": (0.1, 0.5)
    },
    {
        "name": "batch",
        "type": "discrete",
        "domain": (32, 64, 128, 256)
    },
    {
        "name": "l2",
        "type": "continuous",
        "domain": (1e-6, 1e-2)
    }
]

optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective,
    domain=domain,
    acquisition_type="EI"
)

optimizer.run_optimization(max_iter=30)

optimizer.plot_convergence()

plt.savefig("convergence.png")

with open("bayes_opt.txt", "w") as f:
    f.write("Best Hyperparameters\n")
    f.write(str(optimizer.x_opt))
    f.write("\n")
    f.write("Best Objective Value\n")
    f.write(str(optimizer.fx_opt))
