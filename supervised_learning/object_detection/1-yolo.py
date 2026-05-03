#!/usr/bin/env python3
"""
Module for Yolo class with output processing
"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    Uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo class
        """
        self.model = K.models.load_model(model_path, compile=False)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes Darknet outputs into boundary boxes, confidences, and probs
        Args:
            outputs: list of numpy.ndarrays from the model
            image_size: numpy.ndarray [image_height, image_width]
        Returns:
            (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        img_h, img_w = image_size
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, n_anchors, _ = output.shape

            # 1. Extract raw data
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]
            
            # 2. Apply Sigmoid to center offsets and objectness
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            # Center coordinates (bx, by) relative to grid cell
            # Create grid maps
            col = np.tile(np.arange(grid_w), grid_h).reshape(grid_h, grid_w)
            row = np.tile(np.arange(grid_h), grid_w).reshape(grid_w, grid_h).T
            col = col.reshape(grid_h, grid_w, 1)
            row = row.reshape(grid_h, grid_w, 1)

            bx = (sigmoid(t_x) + col) / grid_w
            by = (sigmoid(t_y) + row) / grid_h

            # 3. Box dimensions (bw, bh) relative to input size
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            
            bw = (pw * np.exp(t_w)) / input_w
            bh = (ph * np.exp(t_h)) / input_h

            # 4. Transform to (x1, y1, x2, y2) relative to original image
            x1 = (bx - bw / 2) * img_w
            y1 = (by - bh / 2) * img_h
            x2 = (bx + bw / 2) * img_w
            y2 = (by + bh / 2) * img_h

            # Concatenate to shape (grid_h, grid_w, n_anchors, 4)
            res_box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(res_box)

            # 5. Box Confidence and Class Probabilities
            box_confidences.append(sigmoid(output[..., 4:5]))
            box_class_probs.append(sigmoid(output[..., 5:]))

        return (boxes, box_confidences, box_class_probs)
