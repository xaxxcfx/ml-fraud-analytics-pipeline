import pandas as pd
import os

# Define path
data_path = 'data/creditcard.csv'

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print("--- Dataset Loaded Successfully ---")
    print(f"Shape: {df.shape}")
    print("\n--- Class Distribution ---")
    print(df['Class'].value_counts(normalize=True) * 100)
else:
    print(f"Error: Could not find {data_path}. Please ensure the CSV is in the data folder.")
