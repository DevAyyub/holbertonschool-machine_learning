#!/usr/bin/env python3
"""
This module provides a function to remove entries from a pandas
DataFrame where a specific column has NaN values.
"""


def prune(df):
    """
    Removes any entries where the 'Close' column has NaN values.

    Args:
        df: The input DataFrame.

    Returns:
        The modified DataFrame with NaN values in 'Close' removed.
    """
    return df.dropna(subset=['Close'])
