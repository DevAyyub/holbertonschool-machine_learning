#!/usr/bin/env python3
"""
Module to calculate precision for each class in a confusion matrix
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix
    confusion is a numpy.ndarray of shape (classes, classes)
    row indices = correct labels, column indices = predicted labels
    Returns: a numpy.ndarray of shape (classes,) containing precision
    """
    TP = np.diagonal(confusion)
    predicted_positives = np.sum(confusion, axis=0)
    return TP / predicted_positives
