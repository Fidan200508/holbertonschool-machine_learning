#!/usr/bin/env python3
"""
Module for Yolo class with image loading
"""
import tensorflow.keras as K
import numpy as np
import cv2
import glob
import os


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
        """Processes Darknet outputs into boundary boxes, confidences, and probs"""
        # ... (Method logic from task 1 remains here)
        pass

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filters boxes based on objectness and class scores"""
        # ... (Method logic from task 2 remains here)
        pass

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Suppresses overlapping boxes using Non-Max Suppression"""
        # ... (Method logic from task 3 remains here)
        pass

    @staticmethod
    def load_images(folder_path):
        """
        Loads all images from a specific folder
        Args:
            folder_path: string path to the folder containing images
        Returns:
            (images, image_paths)
        """
        image_paths = glob.glob(folder_path + '/*', recursive=False)
        images = []
        
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
                
        return images, image_paths
