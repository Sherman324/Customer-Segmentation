# Importing libraries
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model/model.pkl')

# Create an instance of FastAPI
app = FastAPI()

# Define the request body for input data (Features or independent variables)
class PredictRequest(BaseModel):
    Spending_Score: float
    Family_Size: float

# Define the prediction endpoint
@app.post("/predict")
def predict(request: PredictRequest):
    # Access the input data from the request object
    data = np.array([[request.Spending_Score, request.Family_Size]])
    
    # Get the prediction from the model
    prediction = model.predict(data)  

    # Map the prediction to segmentation labels
    Segmentation_map = {1: "A", 2: "B", 3: "C", 4: "D"}
    return {"prediction": Segmentation_map[int(prediction[0])]}

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Classification API"}


