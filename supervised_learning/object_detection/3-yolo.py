#!/usr/bin/env python3
"""
Module for Yolo class with Non-max Suppression
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
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            box_class = np.argmax(scores, axis=-1)
            box_score = np.max(scores, axis=-1)
            mask = box_score >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(box_class[mask])
            box_scores.append(box_score[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Suppresses overlapping boxes using Non-Max Suppression
        Args:
            filtered_boxes: np.ndarray (?, 4)
            box_classes: np.ndarray (?,)
            box_scores: np.ndarray (?)
        Returns:
            (box_predictions, predicted_box_classes, predicted_box_scores)
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Process NMS per class to avoid suppressing different objects
        for cls in np.unique(box_classes):
            # Extract indices for the current class
            cls_indices = np.where(box_classes == cls)[0]
            
            cls_boxes = filtered_boxes[cls_indices]
            cls_scores = box_scores[cls_indices]
            
            # Sort boxes by score in descending order
            order = np.argsort(cls_scores)[::-1]
            
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                
                if order.size == 1:
                    break
                
                # Calculate Intersection over Union (IOU)
                b1 = cls_boxes[i]
                b2 = cls_boxes[order[1:]]
                
                # Intersection area
                x1 = np.maximum(b1[0], b2[:, 0])
                y1 = np.maximum(b1[1], b2[:, 1])
                x2 = np.minimum(b1[2], b2[:, 2])
                y2 = np.minimum(b1[3], b2[:, 3])
                
                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                intersection = inter_w * inter_h
                
                # Union area
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
                union = area1 + area2 - intersection
                
                iou = intersection / union
                
                # Keep only boxes with IOU less than the threshold
                inds = np.where(iou <= self.nms_t)[0]
                order = order[inds + 1]
                
            box_predictions.append(cls_boxes[keep])
            predicted_box_classes.append(np.full(len(keep), cls))
            predicted_box_scores.append(cls_scores[keep])

        # Combine results from all classes
        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return (box_predictions, predicted_box_classes, predicted_box_scores)
