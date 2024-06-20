# Import necessary libraries
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Bidirectional, LSTM, RepeatVector, TimeDistributed
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import forecast_evaluation  # This is a custom module for forecast evaluation

# Define a function for building Bidirectional LSTM Autoencoder model
def build_blstm_ae_model(input_shape):
    # BLSTM encoder
    encoder_inputs = Input(shape=input_shape)
    encoder = Bidirectional(LSTM(units=200, return_sequences=True, activation='relu', kernel_initializer='he_uniform'))(encoder_inputs)
    encoder = Bidirectional(LSTM(units=100, return_sequences=False, activation='relu', kernel_initializer='he_uniform'))(encoder)

    # BLSTM decoder
    decoder = RepeatVector(input_shape[0])(encoder)
    decoder = Bidirectional(LSTM(units=100, return_sequences=True, activation='relu', kernel_initializer='he_uniform'))(decoder)
    decoder = Bidirectional(LSTM(units=200, return_sequences=True, activation='relu', kernel_initializer='he_uniform'))(decoder)
    decoder_outputs = TimeDistributed(Dense(units=1, activation='linear', kernel_initializer='he_uniform'))(decoder)

    # BLSTM Autoencoder
    blstm_ae_model = Model(encoder_inputs, decoder_outputs)

    # Compile the BLSTM Autoencoder model
    opt = keras.optimizers.Adam(learning_rate=0.001)
    blstm_ae_model.compile(loss='mean_squared_error', optimizer=opt)
    
    return blstm_ae_model

# Define a function for building Bidirectional LSTM model
def build_blstm_model(input_shape):
    # Initialize Sequential model
    blstm_model = Sequential()
    # Add BLSTM layers
    blstm_model.add(Bidirectional(LSTM(units=200, return_sequences=True, activation='relu', kernel_initializer='he_uniform')))
    blstm_model.add(Bidirectional(LSTM(units=100, return_sequences=False, activation='relu', kernel_initializer='he_uniform')))
    # Add Dense layers
    blstm_model.add(Dense(units=50, activation='relu', kernel_initializer='he_uniform'))
    blstm_model.add(Dense(units=1, activation='linear', kernel_initializer='he_uniform'))
    # Compile the BLSTM model
    opt = keras.optimizers.Adam(learning_rate=0.001)
    blstm_model.compile(loss='mean_squared_error', optimizer=opt)
    
    return blstm_model

# Define a function for training and testing the models
def blstm_ae_blstm_model(X, y, epochs=1, file_name='model_prediction.csv'):
    # Get data shape
    batches = X.shape[0]
    timesteps = X.shape[1]
    features = X.shape[2]
    
    # Reshape the input data if needed
    X = X.reshape(batches, timesteps, features)
    
    # Splitting data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)
    
    # Define BLSTM Autoencoder input shape
    input_shape = x_train.shape[1:]
    
    # Build BLSTM Autoencoder model
    blstm_ae_model = build_blstm_ae_model(input_shape)
    
    # Fit BLSTM Autoencoder model on the training data
    print('[INFO]---|| *** Training BLSTM Autoencoder Model...\n')
    blstm_ae_model.fit(x_train, x_train, epochs=epochs, batch_size=16, validation_data=(x_test, y_test), verbose=0)
    print('[INFO]---|| *** BLSTM Autoencoder Model Trained!\n')
    
    # Define BLSTM encoder model without the decoder
    blstm_e_model = Model(inputs=blstm_ae_model.inputs, outputs=blstm_ae_model.layers[2].output)
    
    # Save the BLSTM encoder
    print('[INFO]---|| *** Saving the BLSTM Encoder Model...\n')
    blstm_e_model.save('Models/blstm_e_model.h5')
    print('[INFO]---|| *** BLSTM Encoder Model Saved!\n')
    
    # Extract features using the BLSTM encoder for training
    blstm_e_model_features = blstm_e_model.predict(x_train, verbose=0)
    
    # Reshape the input data for BLSTM
    blstm_e_model_features = blstm_e_model_features.reshape(blstm_e_model_features.shape[0], 1, blstm_e_model_features.shape[1])
    
    # Build BLSTM model
    blstm_model = build_blstm_model(input_shape)
    
    # Fit the BLSTM-E-BLSTM model on the training data
    print('[INFO]---|| *** Training BLSTM-E-BLSTM Model...\n')
    blstm_model.fit(blstm_e_model_features, y_train, epochs=epochs, batch_size=16, verbose=0)
    print('[INFO]---|| *** BLSTM-E-BLSTM Model Trained!\n')
    
    # Save the BLSTM-E-BLSTM model
    print('[INFO]---|| *** Saving BLSTM-E-BLSTM Model...\n')
    blstm_model.save('Models/blstm_model.h5')
    print('[INFO]---|| *** BLSTM-E-BLSTM Model Saved!\n')

    # Extract features using the BLSTM encoder for testing
    blstm_e_model_features = blstm_e_model.predict(x_test, verbose=0)
    
    # Reshape the input data for BLSTM
    blstm_e_model_features = blstm_e_model_features.reshape(blstm_e_model_features.shape[0], 1, blstm_e_model_features.shape[1])

    print('[INFO]---|| *** Testing the BLSTM-E-BLSTM Model...\n')    
    yhat = blstm_model.predict(blstm_e_model_features)
    print('[INFO]---|| *** BLSTM-E-BLSTM Model Testing Completed!\n')

    # Saving predictions to a CSV file
    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': yhat.flatten()})
    df.to_csv(file_name, index=False)
    print("CSV file '{}' created successfully.".format(file_name))

    # Evaluating model predictions
    forecast_evaluation.evaluate_forecasts(y_test, yhat)
