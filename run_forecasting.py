import numpy as np
import pandas as pd
import data_processing # This is a custom module
import blstm_ae_blstm # This is a custom module

# Load the dataset from the 'raw data' directory
dataset = pd.read_csv('raw data/pv_clean.csv', header=0, low_memory=False,
                      infer_datetime_format=True, parse_dates=[0], index_col=['datetime']) 

# Choose data frequency
frequency = 'H'  # Data frequency is set to hourly

# Define the number of time steps for input and output
n_steps_in = 2

# Number of epochs for training
epochs = 100

# Define the filename for the CSV file to save predictions
file_name = 'hourly_actual_predicted_values.csv'

# Clean the data
cleaned_data = data_processing.clean_data(dataset, frequency)

# Convert data into input/output sequences
X, y = data_processing.split_sequences(cleaned_data, n_steps_in)

# Model training and testing
blstm_ae_blstm.blstm_ae_blstm_model(X, y, epochs, file_name)
