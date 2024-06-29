from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Initialize FastAPI app
app = FastAPI()

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

# Define input data model
class PredictionInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI ML Model API"}

# Prediction endpoint
@app.post("/predict/")
def predict(input_data: PredictionInput):
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([input_data.dict()])

        # Perform prediction
        prediction = model.predict(data)
        probability = model.predict_proba(data).max()

        return {"prediction": int(prediction[0]), "probability": probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
