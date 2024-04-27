import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the training and test data
train_data = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\Projects\Exoplanet_Identification\datasets\exoTrain.csv")
test_data = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\Projects\Exoplanet_Identification\datasets\exoTest.csv")

# Separate features (X) and labels (y)
X_train = train_data.drop("LABEL", axis=1).values
y_train = train_data["LABEL"].values
X_test = test_data.drop("LABEL", axis=1).values
y_test = test_data["LABEL"].values

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data for LSTM input (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('\nTest accuracy:', test_acc)

# Save the model
model.save("exoplanet_identification_model_lstm.h5")
