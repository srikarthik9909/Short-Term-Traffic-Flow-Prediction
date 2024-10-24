import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Load and preprocess the data
folder_paths = ["Station - 1","Station - 2","Station - 3","Station - 4"]

dfs_per_folder = []
for idx, folder_path in enumerate(folder_paths, start=1):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f != 'combined.csv']
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    dfs = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df.columns = [f"{col}_{idx}" for col in df.columns]
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    dfs_per_folder.append(combined_df)

final_combined = pd.concat(dfs_per_folder, axis=1)
final_combined_df = final_combined.dropna()

# Scaling the specific flow column
column1 = ["Station - 3"]
station_scaler = MinMaxScaler()
final_combined_df[column1] = station_scaler.fit_transform(final_combined_df[column1])

# Scaling the multiple flow columns
flow_columns = ['5 Minutes_1',"Station - 1","Station - 2","Station - 3","Station - 4"]
flows_df = final_combined_df[flow_columns].copy()
flows_df.rename(columns={'5 Minutes_1': '5 Minutes'}, inplace=True)

columns_to_scale = ["Station - 1","Station - 2","Station - 3","Station - 4"]
scaler = MinMaxScaler()
flows_df[columns_to_scale] = scaler.fit_transform(flows_df[columns_to_scale])

# Function to create sequences
def create_sequences(column_data, time_steps):
    X, Y = [], []
    for i in range(len(column_data) - time_steps):
        X.append(column_data[i:i + time_steps])
        Y.append(column_data[i + time_steps])
    return np.array(X), np.array(Y)

column_data = flows_df[columns_to_scale].values
time_steps = 15

cnn_X, _ = create_sequences(column_data, time_steps)
station1_data = column_data[:, 0].reshape(-1, 1)
lstm_X, y = create_sequences(station1_data, time_steps)

train_size = int(len(cnn_X) * 0.85)
X_test_cnn = cnn_X[train_size:]
X_test_lstm = lstm_X[train_size:]
y_test = y[train_size:]

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="Conv_LSTM_Model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

predicted_values = []
true_values = []

# Make predictions using the TFLite model
for i in range(len(X_test_lstm)):
    cnn_input_data = np.expand_dims(X_test_cnn[i], axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], cnn_input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_values.append(output_data[0][0])
    true_values.append(y_test[i])

predicted_value = np.array(predicted_values)
true_value = np.array(true_values)

# Rescale the predictions and actual values
predictions_rescaled = station_scaler.inverse_transform(predicted_value.reshape(-1, 1)).flatten()
y_test_rescaled = station_scaler.inverse_transform(true_value.reshape(-1, 1)).flatten()

# Display results
results = pd.DataFrame(data={'Predictions': predictions_rescaled, 'Actuals': y_test_rescaled})
print("============================================")
print("TFLite Model Results:")
print("============================================")
print(results)


# Calculate and print evaluation metrics
rmse = math.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
mape = mean_absolute_percentage_error(y_test_rescaled, predictions_rescaled)
print("============================================")
print("TFLite Model RMSE:", rmse)
print("TFLite Model MAE:", mae)
print("TFLite Model MAPE:", mape)