#!/usr/bin/env python3
"""
Module that initializes a Yolo object detection model.
"""
import tensorflow.keras as K
import numpy as np


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

    def process_outputs(self, outputs, image_size):
        """
        Processes outputs of the Darknet model.

        Args:
            outputs (list): list of numpy.ndarrays containing the
                predictions from the Darknet model for a single image.
            image_size (numpy.ndarray): array containing the image's
                original size [image_height, image_width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        img_h, img_w = image_size[0], image_size[1]
        model_h = int(self.model.input.shape[1])
        model_w = int(self.model.input.shape[2])

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]
            box_conf = output[..., 4:5]
            classes = output[..., 5:]

            box_confidences.append(1 / (1 + np.exp(-box_conf)))
            box_class_probs.append(1 / (1 + np.exp(-classes)))

            cx, cy = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
            cx = np.expand_dims(cx, axis=-1)
            cy = np.expand_dims(cy, axis=-1)

            b_x = (1 / (1 + np.exp(-t_x)) + cx) / grid_w
            b_y = (1 / (1 + np.exp(-t_y)) + cy) / grid_h

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            b_w = (pw * np.exp(t_w)) / model_w
            b_h = (ph * np.exp(t_h)) / model_h

            x1 = (b_x - (b_w / 2)) * img_w
            y1 = (b_y - (b_h / 2)) * img_h
            x2 = (b_x + (b_w / 2)) * img_w
            y2 = (b_y + (b_h / 2)) * img_h

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))

        return boxes, box_confidences, box_class_probs
