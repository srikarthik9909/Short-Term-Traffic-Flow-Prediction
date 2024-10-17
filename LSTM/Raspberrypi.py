import os
import math
import pandas as pd
import numpy as np
import re 
import tensorflow as tf

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Function to form the data into sequences
def create_sequences(column_data, time_steps):
    X, Y = [], []
    for i in range(len(column_data) - time_steps):
        X.append(column_data[i:i + time_steps])
        Y.append(column_data[i + time_steps])
    return np.array(X), np.array(Y)

# Combinig all csv files on a variabel
directory_path = 'Path to csv file'
dataset = pd.read_csv(directory_path)

# Checking the Null Values in the data
print("<------------------------------------------------ Null Values of the DataSet ------------------------------------------------>")
print(dataset.isnull().sum())
# Initializing the MinMax Scaler on the required data
print("<----------------------------------------------------- Normalized Data ----------------------------------------------------->")
column = ['Required column in the dataset you working on']
scaler = MinMaxScaler()
dataset[column] = scaler.fit_transform(dataset[column])
print(dataset.head())

# Sequence formation as per required shape 
print("<--------------------------------- Sequence Formation and Spliting Test and Training Data --------------------------------->")
column_data = dataset[['Required column in the dataset you working on']].values
time_steps = 15
LSTM_X, y = create_sequences(column_data, time_steps)
print(f"Shape of LSTM_X: {LSTM_X.shape}")
print(f"Shape of y: {y.shape}")

# Loading the TFLite model to make Predictions
print("<--------------------------------------------------- Prediction Values --------------------------------------------------->")
interpreter = tf.lite.Interpreter(model_path='')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
predicted_values = []
true_values = []

for i in range(len(LSTM_X)):
    cnn_input_data = np.expand_dims(LSTM_X[i], axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], cnn_input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_values.append(output_data[0][0])
    true_values.append(y[i])
predicted_values = np.array(predicted_values)
true_values = np.array(true_values)

predicted_values_reshaped = predicted_values.reshape(-1, 1)
predictions_rescaled = scaler.inverse_transform(predicted_values_reshaped).flatten()
y_test_reshaped = true_values.reshape(-1, 1)
y_test_rescaled = scaler.inverse_transform(y_test_reshaped).flatten()

results = pd.DataFrame(data={'Predictions': predictions_rescaled, 'Actuals': y_test_rescaled})
print(results)

# Calculating the loss values of the TFLite model
print("<------------------------------------------------ Loss Values of the Model ------------------------------------------------>")
rmse = math.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
mape = mean_absolute_percentage_error(predictions_rescaled, y_test_rescaled)
print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE:", mape)
