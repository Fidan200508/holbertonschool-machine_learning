#!/usr/bin/env python3
"""
Module containing the Yolo class to initialize YOLO v3 object detection
"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    Uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor
        Args:
            model_path: path to where a Darknet Keras model is stored
            classes_path: path to the list of class names used for the model
            class_t: float representing the box score threshold
            nms_t: float representing the IOU threshold for non-max suppression
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
        """
        # Load the Darknet Keras model
        # Using compile=False as we are only using the model for inference
        self.model = K.models.load_model(model_path, compile=False)

        # Load class names from the provided file path
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
