#!/usr/bin/env python3
"""
Contains the function l2_reg_cost
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization
    Args:
        cost: tensor containing the cost of the network without L2 reg
        model: Keras model that includes layers with L2 regularization
    Returns:
        A tensor containing the total cost for each layer of the network
    """
    # model.losses contains the regularization penalties for each layer
    # We add the base cost to each of these individual penalties
    return cost + model.losses
