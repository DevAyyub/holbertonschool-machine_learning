#!/usr/bin/env python3
"""
This module provides a function to create a pandas DataFrame
from a given numpy ndarray.
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray with alphabetical columns.

    Args:
        array (np.ndarray): The numpy array to convert.

    Returns:
        pd.DataFrame: The newly created DataFrame.
    """
    num_cols = array.shape[1]
    columns = [chr(65 + i) for i in range(num_cols)]
    return pd.DataFrame(array, columns=columns)
