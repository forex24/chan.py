# Install necessary libraries
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("autogluon")
install("pandas")
install("fastparquet")
install("matplotlib")
install("seaborn")

# Import necessary libraries
import pandas as pd
import fastparquet as fp
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from autogluon.tabular import TabularPredictor

# Read parquet file
file_path = '/opt/data/eurusd.parquet'
df = pd.read_parquet(file_path, engine='fastparquet')

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Read libsvm file
def read_libsvm(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0])
            features = {}
            for part in parts[1:]:
                index, value = part.split(':')
                features[int(index)] = float(value)
            data.append((label, features))
    return data

# Read external features
feature_data = read_libsvm('feature.libsvm')

# Convert external features to DataFrame
external_features = pd.DataFrame([item[1] for item in feature_data], columns=range(14))

# Read metadata
with open('feature.meta', 'r') as f:
    metadata = json.load(f)

# Map metadata column names to DataFrame
external_features.rename(columns={v: k for k, v in metadata.items()}, inplace=True)

# Merge external features with existing dataset
df = df.join(external_features, how='inner')

# Define functions to calculate technical indicators
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    df['MACD'] = df['close'].ewm(span=short_window, adjust=False).mean() - df['close'].ewm(span=long_window, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    df['MACD_diff'] = df['MACD'] - df['MACD_signal']
    return df['MACD_diff']

def calculate_rsi(df, window=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df['RSI']

def calculate_bollinger_bands(df, window=20, num_std=2):
    df['Bollinger_Middle'] = df['close'].rolling(window=window, min_periods=1).mean()
    df['Bollinger_High'] = df['Bollinger_Middle'] + (df['close'].rolling(window=window, min_periods=1).std() * num_std)
    df['Bollinger_Low'] = df['Bollinger_Middle'] - (df['close'].rolling(window=window, min_periods=1).std() * num_std)
    return df[['Bollinger_High', 'Bollinger_Low']]

def calculate_demark(df, period=9):
    df['DeMark'] = (df['high'].rolling(window=period, min_periods=1).max() + df['low'].rolling(window=period, min_periods=1).min()) / 2
    return df['DeMark']

# Calculate technical indicators
df['MACD_diff'] = calculate_macd(df)
df['RSI'] = calculate_rsi(df)
bollinger_bands = calculate_bollinger_bands(df)
df['Bollinger_High'] = bollinger_bands['Bollinger_High']
df['Bollinger_Low'] = bollinger_bands['Bollinger_Low']
df['DeMark'] = calculate_demark(df)

# Calculate the next minute's price change
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Drop the last row as it has no target variable
df.dropna(inplace=True)

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

# Separate features and target variable
X_train = train_data.drop(columns=['target'])
y_train = train_data['target']

X_test = test_data.drop(columns=['target'])
y_test = test_data['target']

# Create an AutoML model
predictor = TabularPredictor(label='target').fit(train_data=X_train, label=y_train, time_limit=120)

# View the best models
print(predictor.leaderboard(X_test, y_test, silent=True))

# Predict
y_pred = predictor.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Sell', 'Buy'], yticklabels=['Sell', 'Buy'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred, target_names=['Sell', 'Buy']))

# Feature importance analysis
feature_importance = predictor.feature_importance(X_test, y_test)

# Plot feature importance
plt.figure(figsize=(10, 8))
feature_importance.sort_values(ascending=False).plot(kind='barh')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, y_test, label='Actual')
plt.plot(test_data.index, y_pred, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted')
plt.xlabel('Timestamp')
plt.ylabel('Target')
