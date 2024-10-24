import os
import math
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def create_lstm_sequences(data, time_steps, start_idx):
    X, Y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        Y.append(data[i + time_steps])
    return np.array(X), np.array(Y)

def Day_Wise_Sequence(df, files=13, rows_per_file=2016, rows_target=91):
    reshaped_data = []
    for i in range(files):
        start_idx = i * rows_per_file
        end_idx = start_idx + rows_per_file
        num_rows = end_idx - start_idx
        target_shape = (num_rows // 7, 7)
        reshaped_segment = df[start_idx:end_idx].reshape(target_shape)
        reshaped_data.append(reshaped_segment)
    final_data = np.concatenate(reshaped_data, axis=1)
    return pd.DataFrame(final_data)

def generate_sequences(reshaped_station, time_steps=4, sequence_length=15):
    i = 0
    j = 0
    k = time_steps - 1
    sequences = []
    sequence = []
    while i < reshaped_station.shape[0] - time_steps + 1 and j < reshaped_station.shape[1] - sequence_length + 1:
        sequence.append(reshaped_station[i][0 + j:sequence_length + j])
        if i >= k:
            i = k - time_steps
            j += 1
            sequences.append(sequence)
            sequence = []
        if j >= reshaped_station.shape[1] - sequence_length + 1:
            k += 1
            j = 0
            i = k - time_steps
        i += 1
    return np.array(sequences)

# Converting the CSV files into one file
dataset = pd.read_csv('Path_to_DataSet')
dataset = dataset[::-1]

# Checking the Null Values in the data
print("<------------------------------------------------ Null Values of the DataSet ------------------------------------------------>")
print(dataset.isnull().sum())

# Initializing the MinMax Scaler on the required data
print("<----------------------------------------------------- Normalized Data ----------------------------------------------------->")
column = ['Count']
scaler = MinMaxScaler()
dataset[column] = scaler.fit_transform(dataset[column])
print(dataset.head())

# Sequence formation for the Model as per the conditions
print("<----------------------------------------------- Day Wise Sequence Formation ----------------------------------------------->")
# LSMT Sequence Formation
time_steps = 15
start_idx = 8064
flow_data = dataset['Column Name'].values[start_idx:]
lstm_sequences = flow_data.reshape(-1, 1)  
lstm_X, y = create_lstm_sequences(lstm_sequences, time_steps, start_idx)
print("Shape of LSTM_X Values :",lstm_X.shape)
print("Shape of y Values :",y.shape)

# Sequence formation as per required shape (Null, 288) Day wise Data
reshaped_dataset = Day_Wise_Sequence(dataset['Column Name'].values)
reshaped_dataset = np.array(reshaped_dataset)
reshaped_station = reshaped_dataset.T
print("Shape of Reshaped_Dataset :",reshaped_station.shape) 
print("Values of Reshaped_Datset :",reshaped_station)

# Sequence formation as per required shape (Null,4,15) for Parallel LSTM
sequences = generate_sequences(reshaped_station)
print("Shape of Parllel LSTM Sequence :",sequences.shape)

# Splitting the data into Training and Testing data
min_sequences = min(len(lstm_X), len(sequences))

lstm_sequences = lstm_sequences[:min_sequences]
sequences = sequences[:min_sequences]
sequences = np.array(sequences)
split_idx = int(0.8 * min_sequences)

X_short_term_train = lstm_X[:split_idx]
X_short_term_test = lstm_X[split_idx:]
X_historical_train = sequences[:split_idx]
X_historical_test = sequences[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"X_short_term_train shape: {X_short_term_train.shape}")
print(f"X_short_term_test shape: {X_short_term_test.shape}")
print(f"X_historical_train shape: {X_historical_train.shape}")
print(f"X_historical_test shape: {X_historical_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Loading the TFLite model to make Predictions
print("<--------------------------------------------------- Prediction Values --------------------------------------------------->")
interpreter = tf.lite.Interpreter(model_path='Path_to_TFLite_Model')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
predicted_value = []
true_values = []
for i in range(len(X_short_term_test)):
    input_data_1 = np.expand_dims(X_short_term_test[i], axis=0).astype(np.float32)
    input_data_2 = np.expand_dims(X_historical_test[i], axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data_2)
    interpreter.set_tensor(input_details[1]['index'], input_data_1)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_value.append(output_data[0][0])
    true_values.append(y_test[i])
predicted_value = np.array(predicted_value).reshape(-1, 1)

tflite_predictions_rescaled = scaler.inverse_transform(predicted_value).flatten()
tflite_y_test_rescaled = scaler.inverse_transform(true_values).flatten()

tflite_min_length = min(len(tflite_predictions_rescaled), len(tflite_y_test_rescaled))
tflite_predictions_rescale = tflite_predictions_rescaled[:tflite_min_length]
tflite_y_test_rescale = tflite_y_test_rescaled[:tflite_min_length]

tflite_results = pd.DataFrame(data={'Predictions': tflite_predictions_rescale, 'Actuals': tflite_y_test_rescale})
print(tflite_results)

# Calculating the loss values of the TFLite model
print("<------------------------------------------------ Loss Values of the Model ------------------------------------------------>")
rmse = math.sqrt(mean_squared_error(tflite_y_test_rescale, tflite_predictions_rescale))
mae = mean_absolute_error(tflite_y_test_rescale, tflite_predictions_rescale)
mape = mean_absolute_percentage_error(tflite_predictions_rescale, tflite_y_test_rescale)
print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE:", mape)