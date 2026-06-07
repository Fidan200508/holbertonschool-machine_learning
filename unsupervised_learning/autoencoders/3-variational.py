#!/usr/bin/env python3
"""
Variational Autoencoder
"""

import tensorflow.keras as keras
from tensorflow.keras import backend as K


def sampling(args):
    """
    Reparameterization trick.
    """
    mu, log_sig = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    epsilon = K.random_normal(shape=(batch, dim))

    return mu + K.exp(log_sig / 2) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.
    """

    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mu = keras.layers.Dense(latent_dims)(x)
    log_sig = keras.layers.Dense(latent_dims)(x)

    latent = keras.layers.Lambda(sampling)([mu, log_sig])

    encoder = keras.Model(encoder_input, [latent, mu, log_sig])

    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(decoder_input, decoder_output)

    auto_input = keras.Input(shape=(input_dims,))
    z, mu, log_sig = encoder(auto_input)
    reconstructed = decoder(z)

    auto = keras.Model(auto_input, reconstructed)

    reconstruction_loss = keras.losses.binary_crossentropy(
        auto_input, reconstructed
    )
    reconstruction_loss *= input_dims

    kl_loss = 1 + log_sig - K.square(mu) - K.exp(log_sig)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)

    auto.compile(optimizer='adam')

    return encoder, decoder, auto
