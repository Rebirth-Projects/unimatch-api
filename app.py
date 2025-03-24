from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import traceback  # Import for better error logging

# Load the trained model
model = joblib.load("xgb_model.pkl")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/")
def home():
    return "Welcome to the Course Prediction API!"

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"message": "Allowed"}), 200  # Handle preflight requests

    try:
        # Log request JSON
        data = request.get_json()
        print("Received JSON:", data)

        # Validate input format
        if not data or "features" not in data or not isinstance(data["features"], list):
            return jsonify({"error": "Invalid input: 'features' must be a list"}), 400

        # Convert features into numpy array
        features = np.array(data["features"]).reshape(1, -1)
        print("Processed features:", features)

        # Make prediction
        prediction = model.predict(features)[0]
        print("Prediction result:", prediction)

        return jsonify({"prediction": int(prediction)}), 200

    except Exception as e:
        print("Error during prediction:", str(e))
        print(traceback.format_exc())  # Log full error traceback
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Get port from environment or default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)
