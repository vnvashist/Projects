import yfinance as yf
import pandas as pd
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Download data
data = yf.download("AAPL", start="1981-01-01", end="2024-11-02")

close_prices = data['Close'].squeeze()

# Calculate technical indicators
data['SMA_20'] = SMAIndicator(close=close_prices, window=20).sma_indicator()
data['SMA_100'] = SMAIndicator(close=close_prices, window=100).sma_indicator()
data['RSI'] = RSIIndicator(close=close_prices, window=14).rsi()
macd = MACD(close=close_prices)
data['MACD'] = macd.macd()
data['MACD_signal'] = macd.macd_signal()
# Add Lagged Features
data['Close_lag1'] = data['Close'].shift(1)
data['Volume_lag1'] = data['Volume'].shift(1)
data['Price_Return'] = data['Close'].pct_change()

# Drop any rows with NaN values created by indicators and lagging
data.dropna(inplace=True)

# Select relevant columns for LSTM model
features = ['Close', 'SMA_20', 'SMA_100', 'RSI', 'MACD', 'MACD_signal', 'Close_lag1', 'Volume_lag1', 'Price_Return']
data = data[features]

# Normalize features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences for LSTM
sequence_length = 60
X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 0])  # Predicting 'Close' price

X, y = np.array(X), np.array(y)

# Split data into train and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=25),
    Dropout(0.2),
    Dense(units=1)
])

# Compile model with Adam optimizer and mean squared error loss
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

from tensorflow.keras.callbacks import EarlyStopping

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Predict on test data
predicted = model.predict(X_test)

# Reverse the scaling to get actual prices
predicted_prices = scaler.inverse_transform(np.concatenate([predicted, X_test[:, -1, 1:]], axis=1))[:, 0]
actual_prices = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), X_test[:, -1, 1:]], axis=1))[:, 0]

# Plotting the results
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(actual_prices, color='black', label="Actual Prices")
plt.plot(predicted_prices, color='blue', label="Predicted Prices")
plt.title("Apple Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()

