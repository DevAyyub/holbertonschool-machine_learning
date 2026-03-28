#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove the Weighted_Price column
df = df.drop(columns=['Weighted_Price'])

# Rename the Timestamp column to Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert the timestamp values to datetime values
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index the data frame on Date
df = df.set_index('Date')

# Fill missing values in Close with the previous row's value
df['Close'] = df['Close'].ffill()

# Fill missing values in High, Low, Open with the Close value
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])

# Fill missing values in Volume columns with 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Filter the data from 2017 and beyond
df = df.loc['2017':]

# Group by day and apply aggregation functions
df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Print the transformed dataframe
print(df)

# Plot the data
df.plot()
plt.show()
