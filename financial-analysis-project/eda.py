import pandas as pd
import numpy as np

# Load the data
aapl = pd.read_csv('./data/aapl.csv')

# Handle missing values
aapl.dropna(inplace=True)

print(aapl['Close'].mean())
print(aapl['Close'].median())
print(aapl['Close'].std())