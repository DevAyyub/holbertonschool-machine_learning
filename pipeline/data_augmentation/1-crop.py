#!/usr/bin/env python3
"""
Module that contains the function crop_image
"""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.

    Args:
        image: A 3D tf.Tensor containing the image to crop.
        size: A tuple containing the size of the crop.

    Returns:
        The cropped image.
    """
    return tf.image.random_crop(image, size)
