#!/usr/bin/env python3
"""
This module provides a function to compute descriptive statistics
for all columns in a pandas DataFrame except the Timestamp column.
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp.

    Args:
        df: The input DataFrame.

    Returns:
        A new DataFrame containing the descriptive statistics.
    """
    # Drop the Timestamp column and compute descriptive statistics
    return df.drop(columns=['Timestamp']).describe()
