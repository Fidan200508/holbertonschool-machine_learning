#!/usr/bin/env python3
"""
Transfer Learning script to classify CIFAR-10 using Keras Applications
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the data for the model
    - X: numpy.ndarray of shape (m, 32, 32, 3)
    - Y: numpy.ndarray of shape (m,)
    Returns: X_p, Y_p
    """
    # Use the specific preprocess_input for the chosen application (ResNet50)
    X_p = K.applications.resnet50.preprocess_input(X)
    
    # Convert labels to one-hot encoding
    Y_p = K.utils.to_categorical(Y, 10)
    
    return X_p, Y_p


def train_model():
    """
    Trains a ResNet50 model on CIFAR-10
    """
    # Load CIFAR-10 data
    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()

    # Preprocess data
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_valid_p, Y_valid_p = preprocess_data(X_valid, Y_valid)

    # Input tensor
    inputs = K.Input(shape=(32, 32, 3))

    # Lambda layer to resize images to the size ResNet50 expects (min 197x197)
    # Using 224x224 is standard for ResNet
    input_resized = K.layers.Lambda(
        lambda x: K.backend.resize_images(x, 224//32, 224//32, "channels_last")
    )(inputs)

    # Load ResNet50 base model, excluding the top dense layers
    base_model = K.applications.ResNet50(weights='imagenet',
                                         include_top=False,
                                         input_tensor=input_resized)

    # Freeze the base model layers
    base_model.trainable = False

    # Add custom layers on top
    x = base_model.output
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(512, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    x = K.layers.Dense(256, activation='relu')(x)
    predictions = K.layers.Dense(10, activation='softmax')(x)

    # Compile the final model
    model = K.Model(inputs=inputs, outputs=predictions)
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    # Note: Increasing batch size helps speed up training when using upsampling
    model.fit(X_train_p, Y_train_p,
              batch_size=128,
              epochs=5,
              validation_data=(X_valid_p, Y_valid_p),
              verbose=1)

    # Save the model
    model.save('cifar10.h5')


if __name__ == "__main__":
    train_model()
