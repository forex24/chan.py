# Install necessary libraries
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("fastparquet")
install("pandas")
install("scikit-learn")
install("auto-sklearn")
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
import autosklearn.classification

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

# Define a function to extract features from different timeframes
def extract_features_from_timeframe(df, timeframe):
    df_resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'MACD_diff': 'last',
        'RSI': 'last',
        'Bollinger_High': 'last',
        'Bollinger_Low': 'last',
        'DeMark': 'last'
    })
    return df_resampled

# Define timeframes
timeframes = ['3T', '5T', '15T', '30T', '1H', '2H', '4H', '1D']

# Extract features from different timeframes
features = {}
for tf in timeframes:
    features[tf] = extract_features_from_timeframe(df, tf)

# Merge features
merged_features = pd.concat(features.values(), axis=1, keys=features.keys())

# Calculate the next minute's price change
merged_features['target'] = (merged_features['3T']['close'].shift(-1) > merged_features['3T']['close']).astype(int)

# Drop the last row as it has no target variable
merged_features.dropna(inplace=True)

# Split data into training and testing sets
train_size = int(len(merged_features) * 0.8)
train_data = merged_features[:train_size]
test_data = merged_features[train_size:]

# Separate features and target variable
X_train = train_data.drop(columns=['target'])
y_train = train_data['target']

X_test = test_data.drop(columns=['target'])
y_test = test_data['target']

# Create an AutoML model
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,  # Set time limit
    per_run_time_limit=30,        # Set time limit per model run
    tmp_folder='/tmp/autosklearn_classification_tmp',
    output_folder='/tmp/autosklearn_classification_out'
)

# Train the model
automl.fit(X_train, y_train)

# View the best models
print(automl.leaderboard())

# Predict
y_pred = automl.predict(X_test)

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

# Calculate feature importance
result = permutation_importance(automl, X_test, y_test, n_repeats=10, random_state=42)

# Get feature importance
feature_importances = pd.Series(result.importances_mean, index=X_test.columns)

# Plot feature importance
plt.figure(figsize=(10, 8))
feature_importances.sort_values(ascending=False).plot(kind='barh')
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
plt.show()

# Save the model
import joblib
joblib.dump(automl, 'eurusd_classification_model.pkl')

# Load the model
loaded_model = joblib.load('eurusd_classification_model.pkl')

# Use the loaded model for prediction
y_pred_loaded = loaded_model.predict(X_test)
