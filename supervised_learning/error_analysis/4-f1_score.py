#!/usr/bin/env python3
"""
Module to calculate the F1 score for each class in a confusion matrix
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix
    confusion is a numpy.ndarray of shape (classes, classes)
    row indices = correct labels, column indices = predicted labels
    Returns: a numpy.ndarray of shape (classes,) containing the F1 score
    """
    prec = precision(confusion)
    sens = sensitivity(confusion)
    return 2 * (prec * sens) / (prec + sens)
