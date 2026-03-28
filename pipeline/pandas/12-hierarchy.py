#!/usr/bin/env python3
"""
This module provides a function to concatenate two DataFrames
with a rearranged MultiIndex and specific time slicing.
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenates two dataframes for a specific time range,
    setting Timestamp as the first level of a MultiIndex,
    and sorts them chronologically.

    Args:
        df1: The first DataFrame (coinbase).
        df2: The second DataFrame (bitstamp).

    Returns:
        The concatenated and rearranged DataFrame.
    """
    # Set Timestamp as index for both DataFrames
    df1 = index(df1)
    df2 = index(df2)

    # Slice both DataFrames to the specified time window
    df1_sliced = df1.loc[1417411980:1417417980]
    df2_sliced = df2.loc[1417411980:1417417980]

    # Concatenate df2 (bitstamp) and df1 (coinbase) with keys
    df = pd.concat([df2_sliced, df1_sliced], keys=['bitstamp', 'coinbase'])

    # Rearrange the MultiIndex so Timestamp is the first level
    df = df.swaplevel(0, 1)

    # Sort the index to ensure chronological order
    df = df.sort_index()

    return df
