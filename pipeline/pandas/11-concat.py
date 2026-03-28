#!/usr/bin/env python3
"""
This module provides a function to concatenate two pandas
DataFrames with specific slicing and multi-level indexing.
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Indexes two dataframes on Timestamp, slices df2 up to a specific
    timestamp, concatenates them, and adds hierarchical keys.

    Args:
        df1: The first DataFrame (coinbase).
        df2: The second DataFrame (bitstamp).

    Returns:
        The concatenated DataFrame.
    """
    # Set Timestamp as index for both DataFrames
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # Slice df2 up to and including timestamp 1417411920
    df2_sliced = df2_indexed.loc[:1417411920]

    # Concatenate df2 (bitstamp) on top of df1 (coinbase) with keys
    return pd.concat([df2_sliced, df1_indexed], keys=['bitstamp', 'coinbase'])
