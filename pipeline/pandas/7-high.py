#!/usr/bin/env python3
"""
This module provides a function to sort a pandas DataFrame
by the 'High' price in descending order.
"""


def high(df):
    """
    Sorts a DataFrame by the 'High' column in descending order.

    Args:
        df: The input DataFrame.

    Returns:
        The sorted DataFrame.
    """
    return df.sort_values(by='High', ascending=False)
