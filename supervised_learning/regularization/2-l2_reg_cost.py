#!/usr/bin/env python3
"""L2 regularization cost for a Keras model"""
import tensorflow as tf

def l2_reg_cost(cost, model):
    """
    Computes the cost of a neural network including L2 regularization.

    cost: tensor containing the base cost (without regularization)
    model: Keras model with layers that may include L2 regularization

    Returns: tensor containing total cost including L2 for each layer
    """
    l2_loss = tf.add_n([tf.nn.l2_loss(w) for layer in model.layers 
                        for w in layer.trainable_weights 
                        if 'kernel' in w.name])
    total_cost = cost + l2_loss
    return total_cost
