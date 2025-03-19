import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam

# Define file path
file_path = r"C:\Users\adiap\OneDrive\Documents\Gen Ai\Ass-2\stock.csv"

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' not found. Please check the path.")

# Load dataset
df = pd.read_csv(file_path)

# Trim spaces from column names
df.columns = df.columns.str.strip()

# Display column names for debugging
print("Cleaned Columns:", df.columns)

# Ensure 'Close' column exists
if 'Close' not in df.columns:
    raise KeyError(f"Column 'Close' not found in the dataset. Available columns: {df.columns}")

# Use only the 'Close' column for prediction
df = df[['Close']]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Function to create input-output pairs using a sliding window
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Set sequence length
seq_length = 50  
X, y = create_sequences(scaled_data, seq_length)

# Split into train and test sets (80% training, 20% testing)
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Reshape for RNN input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Function to create and train a model
def build_and_train_model(model_type):
    model = Sequential()
    
    if model_type == "RNN":
        model.add(SimpleRNN(50, activation='relu', return_sequences=False, input_shape=(seq_length, 1)))
    elif model_type == "LSTM":
        model.add(LSTM(50, activation='relu', return_sequences=False, input_shape=(seq_length, 1)))
    elif model_type == "GRU":
        model.add(GRU(50, activation='relu', return_sequences=False, input_shape=(seq_length, 1)))
    
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    print(f"\nTraining {model_type} model...")
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    return model

# Train models
rnn_model = build_and_train_model("RNN")
lstm_model = build_and_train_model("LSTM")
gru_model = build_and_train_model("GRU")

# Make predictions
rnn_pred = rnn_model.predict(X_test)
lstm_pred = lstm_model.predict(X_test)
gru_pred = gru_model.predict(X_test)

# Inverse transform to original scale
rnn_pred = scaler.inverse_transform(rnn_pred)
lstm_pred = scaler.inverse_transform(lstm_pred)
gru_pred = scaler.inverse_transform(gru_pred)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label="Actual", color="black")
plt.plot(rnn_pred, label="RNN Prediction", linestyle="dashed", color="blue")
plt.plot(lstm_pred, label="LSTM Prediction", linestyle="dashed", color="red")
plt.plot(gru_pred, label="GRU Prediction", linestyle="dashed", color="green")
plt.legend()
plt.title("Time-Series Prediction: RNN vs. LSTM vs. GRU")
plt.show()
