#!/usr/bin/env python3
"""
Module that initializes a Yolo object detection model.
"""
import tensorflow.keras as K
import numpy as np
import cv2
import os


class Yolo:
    """
    Class that uses the Yolo v3 algorithm to perform object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo class constructor.
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
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        img_h, img_w = image_size[0], image_size[1]
        model_w = int(self.model.input.shape[1])
        model_h = int(self.model.input.shape[2])

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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters bounding boxes based on box score thresholds.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)
            mask = class_scores >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-max suppression to filter overlapping bounding boxes.
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            idxs = np.where(box_classes == cls)[0]
            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]
            cls_classes = box_classes[idxs]

            sorted_indices = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[sorted_indices]
            cls_scores = cls_scores[sorted_indices]
            cls_classes = cls_classes[sorted_indices]

            while len(cls_boxes) > 0:
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls_classes[0])
                predicted_box_scores.append(cls_scores[0])

                if len(cls_boxes) == 1:
                    break

                x1 = cls_boxes[0, 0]
                y1 = cls_boxes[0, 1]
                x2 = cls_boxes[0, 2]
                y2 = cls_boxes[0, 3]

                x1_rest = cls_boxes[1:, 0]
                y1_rest = cls_boxes[1:, 1]
                x2_rest = cls_boxes[1:, 2]
                y2_rest = cls_boxes[1:, 3]

                xx1 = np.maximum(x1, x1_rest)
                yy1 = np.maximum(y1, y1_rest)
                xx2 = np.minimum(x2, x2_rest)
                yy2 = np.minimum(y2, y2_rest)

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)

                inter = w * h
                area_keep = (x2 - x1) * (y2 - y1)
                area_rest = (x2_rest - x1_rest) * (y2_rest - y1_rest)
                union = area_keep + area_rest - inter

                iou = inter / union

                below_threshold = np.where(iou <= self.nms_t)[0]

                cls_boxes = cls_boxes[below_threshold + 1]
                cls_scores = cls_scores[below_threshold + 1]
                cls_classes = cls_classes[below_threshold + 1]

        return (np.array(box_predictions),
                np.array(predicted_box_classes),
                np.array(predicted_box_scores))

    @staticmethod
    def load_images(folder_path):
        """
        Loads images from a folder.
        """
        image_paths = []
        images = []

        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            img = cv2.imread(path)

            if img is not None:
                image_paths.append(path)
                images.append(img)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocesses images for the Darknet model.
        """
        pimages = []
        image_shapes = []

        model_w = int(self.model.input.shape[1])
        model_h = int(self.model.input.shape[2])

        for image in images:
            image_shapes.append([image.shape[0], image.shape[1]])
            resized = cv2.resize(image, (model_w, model_h),
                                 interpolation=cv2.INTER_CUBIC)
            pimages.append(resized / 255.0)

        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with all boundary boxes, class names,
        and box scores.

        Args:
            image (numpy.ndarray): Unprocessed image.
            boxes (numpy.ndarray): Boundary boxes for the image.
            box_classes (numpy.ndarray): Class indices for each box.
            box_scores (numpy.ndarray): Box scores for each box.
            file_name (str): File path where original image is stored.
        """
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]

            # Draw blue bounding box: cv2 uses (B, G, R) format
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                          (255, 0, 0), 2)

            # Format text and draw red label 5 pixels above top-left corner
            class_name = self.class_names[box_classes[i]]
            score = box_scores[i]
            label = "{} {:.2f}".format(class_name, score)

            cv2.putText(image, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                        1, cv2.LINE_AA)

        # Display image
        cv2.imshow(file_name, image)

        # Wait for key press
        key = cv2.waitKey(0)

        # If 's' is pressed, save the image
        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')

            # Extract just the filename if a full path is passed
            save_path = os.path.join('detections', os.path.basename(file_name))
            cv2.imwrite(save_path, image)

        cv2.destroyAllWindows()
