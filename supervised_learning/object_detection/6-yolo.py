#!/usr/bin/env python3
"""
Module for Yolo class with box visualization
"""
import tensorflow.keras as K
import numpy as np
import cv2
import os


class Yolo:
    """
    Uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initializes the Yolo class"""
        self.model = K.models.load_model(model_path, compile=False)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    # ... (Previous methods: process_outputs, filter_boxes, etc.)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with all boundary boxes, class names, and scores
        Args:
            image: numpy.ndarray containing an unprocessed image
            boxes: numpy.ndarray containing the boundary boxes
            box_classes: numpy.ndarray containing the class indices
            box_scores: numpy.ndarray containing the box scores
            file_name: the file path where the original image is stored
        """
        # Create a copy to avoid modifying the original image array
        img_draw = image.copy()

        for i in range(len(boxes)):
            # Box coordinates
            x1, y1, x2, y2 = boxes[i].astype(int)
            
            # Label construction: "Class Name Score"
            class_name = self.class_names[box_classes[i]]
            score = box_scores[i]
            label = "{} {:.2f}".format(class_name, score)

            # Draw the box: Blue (BGR: 255, 0, 0), Thickness 2
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Draw the text: Red (BGR: 0, 0, 255), 5px above top-left
            # FONT_HERSHEY_SIMPLEX, Scale 0.5, Thickness 1, LINE_AA
            cv2.putText(img_draw, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                        1, cv2.LINE_AA)

        # Display image
        cv2.imshow(file_name, img_draw)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        
        # If 's' is pressed, save image
        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            cv2.imwrite(os.path.join('detections', file_name), img_draw)
        
        cv2.destroyAllWindows()
