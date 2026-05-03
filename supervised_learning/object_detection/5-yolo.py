#!/usr/bin/env python3
"""
Module for Yolo class with image preprocessing
"""
import tensorflow.keras as K
import numpy as np
import cv2
import glob


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

    def preprocess_images(self, images):
        """
        Preprocesses a list of images for the Darknet model
        Args:
            images: list of images as numpy.ndarrays
        Returns:
            (pimages, image_shapes)
        """
        # Get model input dimensions
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            # Save original shape (height, width)
            image_shapes.append(img.shape[:2])

            # Resize with inter-cubic interpolation
            resized = cv2.resize(img, (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)

            # Rescale pixel values to [0, 1]
            rescaled = resized / 255.0
            pimages.append(rescaled)

        # Convert to numpy arrays
        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    # Placeholder for previous methods to maintain class structure
    def process_outputs(self, outputs, image_size):
        """Processed outputs"""
        pass

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filters boxes"""
        pass

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """NMS"""
        pass

    @staticmethod
    def load_images(folder_path):
        """Loads images"""
        image_paths = glob.glob(folder_path + '/*', recursive=False)
        images = [cv2.imread(path) for path in image_paths if path]
        return [img for img in images if img is not None], image_paths
