#!/usr/bin/env python3
"""
Module for Yolo class with box filtering
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
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        img_h, img_w = image_size
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, n_anchors, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            col = np.tile(np.arange(grid_w), grid_h).reshape(grid_h, grid_w)
            row = np.tile(np.arange(grid_h), grid_w).reshape(grid_w, grid_h).T
            col = col.reshape(grid_h, grid_w, 1)
            row = row.reshape(grid_h, grid_w, 1)

            bx = (sigmoid(t_x) + col) / grid_w
            by = (sigmoid(t_y) + row) / grid_h

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            bw = (pw * np.exp(t_w)) / input_w
            bh = (ph * np.exp(t_h)) / input_h

            x1 = (bx - bw / 2) * img_w
            y1 = (by - bh / 2) * img_h
            x2 = (bx + bw / 2) * img_w
            y2 = (by + bh / 2) * img_h

            res_box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(res_box)
            box_confidences.append(sigmoid(output[..., 4:5]))
            box_class_probs.append(sigmoid(output[..., 5:]))

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes based on objectness and class scores
        Args:
            boxes: list of np.ndarrays (grid_h, grid_w, anchors, 4)
            box_confidences: list of np.ndarrays (grid_h, grid_w, anchors, 1)
            box_class_probs: list of np.ndarrays (grid_h, grid_w, anchors, cls)
        Returns:
            (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            # Step 1: Calculate scores for each box (Confidence * Class Prob)
            # Shape: (grid_h, grid_w, anchors, classes)
            scores = box_confidences[i] * box_class_probs[i]

            # Step 2: Find the index of the highest score (the class)
            # Shape: (grid_h, grid_w, anchors)
            box_class = np.argmax(scores, axis=-1)

            # Step 3: Extract the value of the highest score
            # Shape: (grid_h, grid_w, anchors)
            box_score = np.max(scores, axis=-1)

            # Step 4: Create a mask for scores > threshold
            mask = box_score >= self.class_t

            # Step 5: Apply mask and flatten to (Total_Filtered, ...)
            filtered_boxes.append(boxes[i][mask])
            box_classes.append(box_class[mask])
            box_scores.append(box_score[mask])

        # Concatenate results from all output scales
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return (filtered_boxes, box_classes, box_scores)
