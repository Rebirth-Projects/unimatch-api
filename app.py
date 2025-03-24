from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

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
        if not request.is_json:
            return jsonify({"error": "Invalid input: Expected JSON"}), 400
        
        data = request.get_json()
        if "features" not in data or not isinstance(data["features"], list):
            return jsonify({"error": "Invalid input: 'features' must be a list"}), 400

        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]

        return jsonify({"prediction": int(prediction)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Get port from environment or default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)
