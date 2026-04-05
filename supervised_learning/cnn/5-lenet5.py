#!/usr/bin/env python3
from tensorflow import keras as K

def lenet5(X):
    he_init = K.initializers.HeNormal(seed=0)

    # Layer 1: Conv -> 6 filters, 5x5, same padding, ReLU
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=he_init
    )(X)

    # Layer 2: Max Pool -> 2x2, stride 2
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # Layer 3: Conv -> 16 filters, 5x5, valid padding, ReLU
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=he_init
    )(pool1)

    # Layer 4: Max Pool -> 2x2, stride 2
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # Flatten
    flat = K.layers.Flatten()(pool2)

    # Layer 5: Fully connected -> 120 nodes, ReLU
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=he_init
    )(flat)

    # Layer 6: Fully connected -> 84 nodes, ReLU
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=he_init
    )(fc1)

    # Layer 7: Fully connected -> 10 nodes, softmax
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=he_init
    )(fc2)

    # Build and compile the model
    model = K.Model(inputs=X, outputs=output)
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
