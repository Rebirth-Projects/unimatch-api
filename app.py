from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("xgb_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Course Prediction API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        
        # Make a prediction
        prediction = model.predict(features)[0]

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
