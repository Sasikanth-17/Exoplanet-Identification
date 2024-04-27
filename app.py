from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = load_model("exoplanet_identification_model_lstm.h5")

# Load the standard scaler for feature scaling
scaler = StandardScaler()
# Load the column names for feature scaling
column_names = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\Projects\Exoplanet_Identification\datasets\exoTest.csv").drop("LABEL", axis=1).columns

@app.route('/')
def home():
    return render_template('index.html', column_names=column_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input data from the form
        features = [float(request.form[column]) for column in column_names]
        # Scale the input features
        scaled_features = scaler.transform(np.array(features).reshape(1, -1))
        # Reshape the input data for model prediction
        input_data = scaled_features.reshape(1, scaled_features.shape[1], 1)
        # Make prediction using the loaded model
        prediction = model.predict(input_data)
        # Determine the prediction label
        if prediction[0][0] >= 0.5:
            result = 'Exoplanet Detected'
        else:
            result = 'No Exoplanet Detected'
        return render_template('output.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
