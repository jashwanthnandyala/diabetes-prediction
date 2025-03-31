import numpy as np
import joblib
import os
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model_path = "model/diabetes_model.pkl"
scaler_path = "model/scaler.pkl"

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    raise FileNotFoundError("Model or Scaler file not found! Run train_model.py first.")

# Home route (renders HTML page)
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route (API for model inference)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.json
        input_data = np.array(data["features"]).reshape(1, -1)

        # Standardize input data
        std_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(std_data)[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"

        return jsonify({"prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
