# main.py
"""
This is the main file for the FastAPI application.
It defines the API endpoints for the CampusRide Fare Predictor.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import joblib
import pandas as pd

# --- 1. Initialize the FastAPI App ---
# We add a title and description for the auto-generated docs.
app = FastAPI(
    title="CampusRide Fare Predictor API",
    description="An API to predict ride fares for the CampusRide App based on distance, duration, and time of day."
)

# --- 2. Load the Trained Model and Columns ---
# These files were created by our 'train_model.py' script.
# They are loaded once when the application starts.
try:
    model = joblib.load('ride_fare_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    print("Model and columns loaded successfully.")
except FileNotFoundError:
    print("Error: Model files not found. Make sure 'ride_fare_model.pkl' and 'model_columns.pkl' are in the same directory.")
    model = None
    model_columns = None

# --- 3. Define the Input Data Structure using Pydantic ---
# This ensures that any request sent to our endpoint has the correct data types.
# We use an Enum to restrict 'timeOfDay' to the values our model knows.
class TimeOfDay(str, Enum):
    morning = 'Morning'
    afternoon = 'Afternoon'
    evening = 'Evening'
    night = 'Night'

class Ride(BaseModel):
    distance: float
    durationOfRide: float
    timeOfDay: TimeOfDay
    
    # You can add example data for the documentation
    class Config:
        json_schema_extra = {
            "example": {
                "distance": 8.5,
                "durationOfRide": 20,
                "timeOfDay": "Evening"
            }
        }

# --- 4. Create the API Endpoints ---

@app.get("/")
def read_root():
    """A welcome message for the root endpoint."""
    return {"message": "Welcome to the CampusRide Fare Predictor API. Go to /docs for the interactive API documentation."}

@app.post("/predict")
def predict_fare(ride: Ride):
    """
    Predicts the fare for a ride based on its details.

    - **distance**: The distance of the ride in kilometers.
    - **durationOfRide**: The duration of the ride in minutes.
    - **timeOfDay**: The time of day ('Morning', 'Afternoon', 'Evening', 'Night').

    Returns the predicted fare as a JSON object.
    """
    if model is None or model_columns is None:
        return {"error": "Model is not loaded. Please check server logs."}

    # Convert the incoming Pydantic object to a dictionary
    data = ride.model_dump()

    # Create a pandas DataFrame from the single ride data
    input_df = pd.DataFrame([data])

    # --- Preprocessing the input data to match the model's training data ---
    # 1. One-Hot Encode the 'timeOfDay' feature. This converts text to numbers.
    input_df_processed = pd.get_dummies(input_df, columns=['timeOfDay'])

    # 2. Align the columns with the model's training columns.
    # This is a CRUCIAL step! It ensures the API's input has the exact same columns
    # (in the same order) as the data the model was trained on.
    # We create a new DataFrame with the expected columns and fill it with our input data.
    aligned_df = pd.DataFrame(columns=model_columns, index=[0]).fillna(0)

    for col in input_df_processed.columns:
        if col in aligned_df.columns:
            aligned_df[col] = input_df_processed[col].values

    # --- Make a prediction ---
    try:
        prediction = model.predict(aligned_df)
        predicted_fare = prediction[0]
    except Exception as e:
        return {"error": f"Failed to make prediction. Error: {e}"}

    # Return the prediction as a JSON response, rounded to 2 decimal places.
    return {"predicted_fare": round(predicted_fare, 2)}