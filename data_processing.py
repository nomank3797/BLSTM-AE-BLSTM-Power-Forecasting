import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import nan, isnan

# Fill missing values with a value at the same time one day ago
def fill_missing(values):
    one_day = 24
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if isnan(values[row, col]):
                values[row, col] = values[row - one_day, col]

# Normalize data	
def normalize_data(values):
    # Create scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit scaler on data
    scaler.fit(values)
    # Apply transform
    normalized = scaler.transform(values)
    return normalized

# Clean-up data	
def clean_data(data, frequency='M'):
    # Mark all missing values
    data.replace('?', nan, inplace=True)
    # Make dataset numeric
    data = data.astype('float32')
    # Fill missing values
    fill_missing(data.values)
    # Resample data
    #resample_groups = data.resample(frequency)
    #resample_data = resample_groups.mean()
    # Moving average
    rolling = data.rolling(window=3)
    rolling_mean = rolling.mean()
    # Drop NaN 
    rolling_mean = rolling_mean.dropna()
    # Normalize data
    normalized_data = normalize_data(rolling_mean.values)
    return normalized_data
 
# Split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in=1, n_steps_out=1):
    X, y = list(), list()
    for i in range(len(sequences)):
        # Find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # Check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
