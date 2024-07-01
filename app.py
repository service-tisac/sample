from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)

# Load the Iris dataset and train a simple Logistic Regression model
iris = load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the model using joblib
joblib.dump(model, "model.pkl")

# Load the model
model = joblib.load("model.pkl")

# Root endpoint
@app.route("/", methods=["GET"])
def read_root():
    return jsonify({"message": "Welcome to the Flask ML Model API"})

# Prediction endpoint
@app.route("/predict/", methods=["POST"])
def predict():
    try:
        # Parse input data
        input_data = request.json
        feature1 = input_data.get("feature1")
        feature2 = input_data.get("feature2")
        feature3 = input_data.get("feature3")
        feature4 = input_data.get("feature4")

        # Check if all features are provided
        if None in [feature1, feature2, feature3, feature4]:
            return jsonify({"error": "All features must be provided"}), 400

        # Convert input data to DataFrame
        data = pd.DataFrame([[feature1, feature2, feature3, feature4]],
                            columns=["feature1", "feature2", "feature3", "feature4"])

        # Perform prediction
        prediction = model.predict(data)
        probability = model.predict_proba(data).max()

        return jsonify({"prediction": int(prediction[0]), "probability": probability})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# Run the application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
