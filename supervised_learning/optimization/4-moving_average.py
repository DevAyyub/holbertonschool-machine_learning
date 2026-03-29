#!/usr/bin/env python3
"""
Module to calculate the weighted moving average with bias correction
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set
    Args:
        data: list of data to calculate the moving average of
        beta: weight used for the moving average
    Returns:
        A list containing the moving averages of data
    """
    moving_avg = []
    vt = 0
    for i in range(len(data)):
        vt = (beta * vt) + ((1 - beta) * data[i])
        bias_correction = 1 - (beta ** (i + 1))
        moving_avg.append(vt / bias_correction)
    return moving_avg
