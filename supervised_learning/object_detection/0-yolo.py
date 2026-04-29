#!/usr/bin/env python3
"""
Module that initializes a Yolo object detection model.
"""
import tensorflow.keras as K


class Yolo:
    """
    Class that uses the Yolo v3 algorithm to perform object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo class constructor.

        Args:
            model_path (str): Path to where a Darknet Keras model is stored.
            classes_path (str): Path to the list of class names used for
                                the Darknet model.
            class_t (float): Box score threshold for initial filtering step.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Shape (outputs, anchor_boxes, 2)
                                     containing all of the anchor boxes.
        """
        self.model = K.models.load_model(model_path)
        
        with open(classes_path, 'r') as file:
            self.class_names = [line.strip() for line in file]
            
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
