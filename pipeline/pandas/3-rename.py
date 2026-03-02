#!/usr/bin/env python3
"""
This module provides a function to rename columns and convert
timestamps in a pandas DataFrame.
"""
import pandas as pd


def rename(df):
    """
    Renames the Timestamp column to Datetime, converts values to datetime,
    and returns only the Datetime and Close columns.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame with 'Datetime' and 'Close'.
    """
    # Rename the column
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # Convert the values to datetime objects (Unix timestamp to datetime)
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # Return only the specified columns
    return df[['Datetime', 'Close']]
