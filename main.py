# main.py
"""
This is the main file for the FastAPI application.
It defines the API endpoints for the CampusRide Fare Predictor.
This version includes CORS middleware to allow web browser access.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware # <--- 1. IMPORT THIS

# --- 1. Initialize the FastAPI App ---
app = FastAPI(
    title="CampusRide Fare Predictor API",
    description="A simple API to predict ride fares for the CampusRide App POC."
)

# --- 2. CONFIGURE CORS ---
# This is the "permission slip" for the browser.
# We are telling the API to allow requests from any origin (*).
# For a real production app, you would list specific domains.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
) # <--- ADD THIS ENTIRE BLOCK

# --- 3. Load the Trained Model and Columns ---
try:
    model = joblib.load('ride_fare_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    print("Model and columns loaded successfully.")
except FileNotFoundError:
    print("Error: Model files not found...")
    model = None
    model_columns = None

# --- 4. Define the Input Data Structure (Pydantic Models) ---
class TimeOfDay(str, Enum):
    morning = 'Morning'
    afternoon = 'Afternoon'
    evening = 'Evening'
    night = 'Night'

class Ride(BaseModel):
    distance: float
    durationOfRide: float
    timeOfDay: TimeOfDay

# --- 5. Create the API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the CampusRide Fare Predictor API. Go to /docs for testing."}

@app.post("/predict")
def predict_fare(ride: Ride):
    """
    Predicts the fare for a ride based on its details.
    """
    if model is None or model_columns is None:
        return {"error": "Model is not loaded. Please check server logs."}

    data = ride.model_dump()
    input_df = pd.DataFrame([data])
    input_df_processed = pd.get_dummies(input_df, columns=['timeOfDay'])
    aligned_df = pd.DataFrame(columns=model_columns, index=[0]).fillna(0)
    for col in input_df_processed.columns:
        if col in aligned_df.columns:
            aligned_df[col] = input_df_processed[col].values

    prediction = model.predict(aligned_df)
    predicted_fare = prediction[0]

    return {"predicted_fare": round(predicted_fare, 2)}