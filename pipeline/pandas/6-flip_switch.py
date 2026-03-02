#!/usr/bin/env python3
"""
This module provides a function to sort a DataFrame in reverse
chronological order and transpose it.
"""


def flip_switch(df):
    """
    Sorts a DataFrame in reverse chronological order and transposes it.

    Args:
        df: The input DataFrame.

    Returns:
        The transformed DataFrame.
    """
    return df.sort_index(ascending=False).T
