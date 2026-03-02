#!/usr/bin/env python3
"""
This module provides a function to clean and fill missing data
in a pandas DataFrame.
"""


def fill(df):
    """
    Removes the Weighted_Price column and fills missing values
    in the remaining columns according to specific rules.

    Args:
        df: The input DataFrame.

    Returns:
        The modified DataFrame.
    """
    # Remove the Weighted_Price column
    df = df.drop(columns=['Weighted_Price'])

    # Fill missing values in Close with the previous row's value
    df['Close'] = df['Close'].ffill()

    # Fill missing values in High, Low, and Open with the Close value
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # Set missing values in Volume columns to 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
