#!/usr/bin/env python3
"""L2 regularization cost per layer for Keras model"""
import tensorflow as tf

def l2_reg_cost(cost, model):
    """
    Computes the cost of a neural network including L2 regularization per layer.

    cost: tensor containing the base cost (without regularization)
    model: Keras model with layers that include L2 regularization

    Returns: tensor with total cost per layer (base + L2 for that layer)
    """
    l2_losses = []
    for layer in model.layers:
        # Collect L2 for each trainable kernel in the layer
        kernel_l2 = sum(tf.nn.l2_loss(w) for w in layer.trainable_weights if 'kernel' in w.name)
        l2_losses.append(kernel_l2)

    # Convert list to tensor and add base cost
    l2_tensor = tf.stack(l2_losses)
    total_cost_per_layer = l2_tensor + cost
    return total_cost_per_layer
