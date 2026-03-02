cat << 'EOF' > 4-array.py
#!/usr/bin/env python3
"""
This module provides a function to extract specific columns and rows
from a pandas DataFrame and convert them into a numpy ndarray.
"""
import pandas as pd


def array(df):
    """
    Selects the last 10 rows of the 'High' and 'Close' columns
    and converts them into a numpy.ndarray.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        numpy.ndarray: The resulting array.
    """
    return df[['High', 'Close']].tail(10).to_numpy()
