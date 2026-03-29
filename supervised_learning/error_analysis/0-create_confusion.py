#!/usr/bin/env python3
"""
Module to create a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix
    labels is a one-hot numpy.ndarray of shape (m, classes)
    logits is a one-hot numpy.ndarray of shape (m, classes)
    Returns: a confusion numpy.ndarray of shape (classes, classes)
    """
    return np.matmul(labels.T, logits)
