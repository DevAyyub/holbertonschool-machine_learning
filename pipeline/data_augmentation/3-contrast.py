#!/usr/bin/env python3
"""
Module that contains the function change_contrast
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Args:
        image: A 3D tf.Tensor representing the input image.
        lower: A float representing the lower bound of the contrast range.
        upper: A float representing the upper bound of the contrast range.

    Returns:
        The contrast-adjusted image.
    """
    return tf.image.random_contrast(image, lower, upper)
