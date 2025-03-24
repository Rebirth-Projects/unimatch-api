from flask import Flask, request, jsonify
import joblib
import numpy as np
import os  # Moved import here

# Load the trained model
model = joblib.load("xgb_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Course Prediction API!"})  # JSON Response

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure request contains JSON
        if not request.is_json:
            return jsonify({"error": "Invalid input: Expected JSON"}), 400

        # Get JSON data from request
        data = request.get_json()

        # Validate input format
        if "features" not in data or not isinstance(data["features"], list):
            return jsonify({"error": "Invalid input: 'features' must be a list"}), 400
        
        # Convert input to numpy array
        features = np.array(data["features"]).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(features)[0]

        return jsonify({"prediction": int(prediction)}), 200  # Ensure JSON output

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Internal Server Error

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get port from environment or default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)
