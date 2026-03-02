#!/usr/bin/env python3
"""
This module provides a function to set a specific column
as the index of a pandas DataFrame.
"""


def index(df):
    """
    Sets the 'Timestamp' column as the index of the dataframe.

    Args:
        df: The input DataFrame.

    Returns:
        The modified DataFrame with 'Timestamp' as the index.
    """
    return df.set_index('Timestamp')
