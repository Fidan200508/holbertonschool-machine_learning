#!/usr/bin/env python3
"""3-projection_block.py"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    a function that builds a projection block as described in
    Deep Residual Learning for Image Recognition (2015)
    :param A_prev: is the output from the previous layer
    :param filters: is a tuple or list containing F11, F3, F12, respectively:
        - F11 is the number of filters in the first 1x1 convolution
        - F3 is the number of filters in the 3x3 convolution
        - F12 is the number of filters in the second 1x1 convolution
    :param s: is the stride to be used in the first convolution in both
    branches of the projection block
    :return: the activated output of the projection block
    """
    F11, F3, F12 = filters
    initializer = K.initializers.HeNormal(seed=0)
    conv_1 = K.layers.Conv2D(filters=F11, kernel_size=1, strides=s,
                             padding='valid', kernel_initializer=initializer)(
        A_prev)
    norm_1 = K.layers.BatchNormalization(axis=3)(conv_1)
    act_1 = K.layers.Activation('relu')(norm_1)
    conv_2 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                             kernel_initializer=initializer)(act_1)
    norm_2 = K.layers.BatchNormalization(axis=3)(conv_2)
    act_2 = K.layers.Activation('relu')(norm_2)
    conv_3 = K.layers.Conv2D(filters=F12, kernel_size=1, padding='valid',
                             kernel_initializer=initializer)(act_2)
    norm_3 = K.layers.BatchNormalization(axis=3)(conv_3)
    conv_s = K.layers.Conv2D(filters=F12, kernel_size=1, strides=s,
                             padding='valid', kernel_initializer=initializer)(
        A_prev)
    norm_s = K.layers.BatchNormalization(axis=3)(conv_s)
    add = K.layers.Add()([norm_3, norm_s])
    return K.layers.Activation('relu')(add)
