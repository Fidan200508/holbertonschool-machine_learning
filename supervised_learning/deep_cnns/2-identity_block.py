#!/usr/bin/env python3
"""2-identity_block.py
"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    a function that builds an identity block as described in
    Deep Residual Learning for Image Recognition (2015)
    :param A_prev: is the output from the previous layer
    :param filters: is a tuple or list containing F11, F3, F12, respectively:
        - F11 is the number of filters in the first 1x1 convolution
        - F3 is the number of filters in the 3x3 convolution
        - F12 is the number of filters in the second 1x1 convolution
    :return: the activated output of the identity block
    """
    F11, F3, F12 = filters
    initializer = K.initializers.HeNormal(seed=0)
    conv_1 = K.layers.Conv2D(filters=F11, kernel_size=1, padding='valid',
                             kernel_initializer=initializer)(A_prev)
    norm_1 = K.layers.BatchNormalization(axis=3)(conv_1)
    act_1 = K.layers.Activation('relu')(norm_1)
    conv_2 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                             kernel_initializer=initializer)(act_1)
    norm_2 = K.layers.BatchNormalization(axis=3)(conv_2)
    act_2 = K.layers.Activation('relu')(norm_2)
    conv_3 = K.layers.Conv2D(filters=F12, kernel_size=1, padding='valid',
                             kernel_initializer=initializer)(act_2)
    norm_3 = K.layers.BatchNormalization(axis=3)(conv_3)
    add = K.layers.Add()([norm_3, A_prev])
    return K.layers.Activation('relu')(add)
