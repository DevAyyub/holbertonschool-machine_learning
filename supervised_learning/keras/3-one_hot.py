#!/usr/bin/env python3
"""
Module to convert label vector to one-hot matrix
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix

    Args:
        labels: label vector to convert
        classes: the total number of classes

    Returns:
        The one-hot matrix
    """
    return K.utils.to_categorical(labels, num_classes=classes)
