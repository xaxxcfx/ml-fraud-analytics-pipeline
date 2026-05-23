import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data/creditcard.csv')

# Scale 'Amount' and 'Time'
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop old columns
df = df.drop(['Amount', 'Time'], axis=1)

# Split the data (using stratify=df['Class'] to keep the balance equal in train/test)
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Preprocessing complete. Training set shape: {X_train.shape}")
print("Data is now scaled and split with stratification.")
