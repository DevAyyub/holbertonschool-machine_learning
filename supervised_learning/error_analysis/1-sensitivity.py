#!/usr/bin/env python3
"""
Module to calculate sensitivity for each class in a confusion matrix
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix
    confusion is a numpy.ndarray of shape (classes, classes)
    row indices = correct labels, column indices = predicted labels
    Returns: a numpy.ndarray of shape (classes,) containing sensitivity
    """
    TP = np.diagonal(confusion)
    actual_positives = np.sum(confusion, axis=1)
    return TP / actual_positives
