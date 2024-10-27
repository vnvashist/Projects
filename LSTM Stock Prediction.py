import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
# Step 1: Download the data
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2024-10-26')

# Step 2: Prepare the data
data = data[['Close']]  # Use only the 'Close' price
data = data.values  # Convert to numpy array

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create training and test datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Function to create the dataset
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Use 60 time steps (you can adjust this)
time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 3: Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Prediction of the next closing price

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Step 5: Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Check the shapes of train_predict and test_predict
print("Train Predictions Shape:", train_predict.shape)
print("Test Predictions Shape:", test_predict.shape)

# Adjust the data slices to match the predictions
train_actual = data[time_step:time_step + len(train_predict)]
test_actual = data[train_size + time_step:train_size + time_step + len(test_predict)]

# Calculate RMSE for train and test
train_rmse = np.sqrt(mean_squared_error(train_actual, train_predict))
test_rmse = np.sqrt(mean_squared_error(test_actual, test_predict))

print(f'Train RMSE: {train_rmse}, Test RMSE: {test_rmse}')

# Step 7: Forecasting
# Specify the number of steps to forecast
forecast_steps = 100  # Number of future steps to predict

# Prepare the input for forecasting
forecast_input = scaled_data[-time_step:].reshape(1, time_step, 1)

# Forecast future values
forecast_output = []
for _ in range(forecast_steps):
    # Make prediction
    next_step = model.predict(forecast_input)
    forecast_output.append(next_step[0, 0])

    # Update the forecast input
    # Shift the input data, add the prediction to the end
    forecast_input = np.append(forecast_input[:, 1:, :], next_step.reshape(1, 1, 1), axis=1)

# Inverse transform the forecasted values
forecast_output = scaler.inverse_transform(np.array(forecast_output).reshape(-1, 1))

# Step 7: Plot the results
plt.figure(figsize=(14, 5))
plt.plot(np.arange(time_step, time_step + len(train_predict)), train_predict, label='Train Predictions', color='green')
plt.plot(np.arange(train_size + (time_step * 2), train_size + (time_step * 2) + len(test_predict)), test_predict, label='Test Predictions',
         color='blue')
plt.plot(data, label='Actual Data', color='red')

# Plotting the forecasted values
forecast_index = np.arange(len(data), len(data) + forecast_steps)
plt.plot(forecast_index, forecast_output, label='Forecast', color='orange')

plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Apple Stock Price Prediction using LSTM with Forecasting")
plt.legend()
plt.show()