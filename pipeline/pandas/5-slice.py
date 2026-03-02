#!/usr/bin/env python3
"""
This module provides a function to extract specific columns
and slice rows from a pandas DataFrame.
"""


def slice(df):
    """
    Extracts the 'High', 'Low', 'Close', and 'Volume_(BTC)' columns
    and selects every 60th row.

    Args:
        df: The input DataFrame.

    Returns:
        The sliced DataFrame.
    """
    columns = ['High', 'Low', 'Close', 'Volume_(BTC)']
    return df[columns].iloc[::60]
