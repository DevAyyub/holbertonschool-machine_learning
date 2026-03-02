#!/usr/bin/env python3
"""
This module creates a pandas DataFrame from a dictionary
and saves it into the variable 'df'.
"""
import pandas as pd

# Define the data dictionary with column labels as keys
data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

# Define the custom row labels
row_labels = ['A', 'B', 'C', 'D']

# Create the DataFrame and assign it to the variable df
df = pd.DataFrame(data, index=row_labels)
