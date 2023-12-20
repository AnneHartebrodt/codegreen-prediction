import re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Read the training dataset
training_set1 = pd.read_csv('CZ-202001010000-202301010000-actual-60.csv')
last_column_df = training_set1[['percentRenewable', 'startTime']].copy()

# Determine the size of the training set for the model
n_97 = int(len(last_column_df) * 0.97)
dataset_97 = last_column_df.iloc[:n_97]

# Extract the last 100 rows for prediction
last_values = dataset_97.tail(100)

def predict(model_name, last_values):
    """
    Predicts the next 48 hours of percent renewable energy based on a pre-trained model.

    Args:
        model_name (str): The name of the pre-trained model file.
        last_values (pd.DataFrame): DataFrame containing the last values of percentRenewable and startTime.

    Returns:
        pd.DataFrame: DataFrame containing the forecast values and timestamps.
    """
    # Extract scaling technique and sequence length from the model name
    last_values_subset = last_values[['percentRenewable', 'startTime']].copy()
    last_values_subset['startTime'] = pd.to_datetime(last_values_subset['startTime'], format='%Y%m%d%H%M')
    
    # Extract the last timestamp from the input data
    last_timestamp = last_values_subset['startTime'].iloc[-1]
    
    # Extract sequence length from the model name
    match = re.search(r'_(\d+).h5', model_name)
    if not match:
        raise ValueError(f"Invalid model name format: {model_name}")
    seq_len_str = match.group(1)
    seq_len = int(seq_len_str)

    # Load the pre-trained model
    model = load_model(model_name)

    # Extract the last (seq_len-1) values from last_values
    last_values = last_values['percentRenewable'].tail(seq_len-1).values.flatten()

    # Initialize the scaler based on the scaling technique
    if 'MinMaxScaler' in model_name:
        scaler = MinMaxScaler()
    elif 'StandardScaler' in model_name:
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported scaling technique in model name: {model_name}")

    # Fit the scaler on the training data
    scaler.fit(last_column_df[['percentRenewable']])

    # List to store the forecast values
    forecast_values = []

    # Generate forecasts for the next 48 hours
    for _ in range(48):
        # Scale the last values
        scaled_last_values = scaler.transform(last_values.reshape(-1, 1))

        # Prepare the input for prediction
        x_pred = scaled_last_values[-(seq_len-1):].reshape(1, (seq_len-1), 1)

        # Predict the next value
        predicted_value = model.predict(x_pred)

        # Inverse transform the predicted value
        predicted_value = scaler.inverse_transform(predicted_value)

        # Append the predicted value to the forecast_values
        forecast_values.append(predicted_value[0][0])

        # Update last_values with the predicted value
        last_values = np.append(last_values, predicted_value)

    # Generate the next 48 timestamps
    forecast_timestamps = pd.date_range(start=last_timestamp, periods=49, freq='H')[1:]

    # Create a DataFrame with forecast values and timestamps
    forecast_df = pd.DataFrame({'Timestamp': forecast_timestamps, 'Forecast': forecast_values})
    return forecast_df

# Example usage:
model_name = 'CZ_MinMaxScaler_model_24.h5'  # You can replace this with the actual model name
forecast_df = predict(model_name, last_values)
print(forecast_df)
