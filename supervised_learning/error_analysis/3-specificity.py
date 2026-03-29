#!/usr/bin/env python3
"""
Module to calculate specificity for each class in a confusion matrix
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix
    confusion is a numpy.ndarray of shape (classes, classes)
    row indices = correct labels, column indices = predicted labels
    Returns: a numpy.ndarray of shape (classes,) containing specificity
    """
    TP = np.diagonal(confusion)
    actual_positives = np.sum(confusion, axis=1)
    predicted_positives = np.sum(confusion, axis=0)
    total = np.sum(confusion)
    
    # TN = Total - (Row Sum + Col Sum - Diagonal)
    # Actual Negatives = Total - Row Sum
    return (total - (actual_positives + predicted_positives - TP)) / (total - actual_positives)
