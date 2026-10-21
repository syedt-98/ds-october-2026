from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Housing Price Prediction API")

# Load the trained deep DNN model and the feature scaler
model = tf.keras.models.load_model("best_model.keras")
scaler = joblib.load("scaler.pkl")

# Define the expected input data structure using Pydantic
class HouseFeatures(BaseModel):
    # Replace these placeholder feature names with your actual dataset feature names!
    feature_1: float
    feature_2: float
    feature_3: float
    # (Add all the numerical features your model expects)

@app.get("/")
def home():
    return {"message": "Welcome to the Housing Price Prediction API!"}

@app.post("/predict")
def predict_price(data: HouseFeatures):
    # 1. Convert incoming JSON data into a numpy array
    input_data = np.array([[
        data.feature_1,
        data.feature_2,
        data.feature_3
        # Match the order of your features here
    ]])
    
    # 2. Scale the input data using the saved scaler
    scaled_data = scaler.transform(input_data)
    
    # 3. Make a prediction using our deep DNN model
    prediction = model.predict(scaled_data)
    
    # 4. Return the predicted price (flattening the array to a single float)
    predicted_price = float(prediction[0][0])
    
    return {"predicted_house_price": predicted_price}