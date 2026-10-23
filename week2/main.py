from typing import Literal

import joblib
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Housing Price Prediction API")

model = tf.keras.models.load_model("best_model.keras")
scaler = joblib.load("scaler.pkl")

FEATURES = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "ocean_proximity_<1H OCEAN", "ocean_proximity_INLAND", "ocean_proximity_ISLAND",
    "ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN",
]


class HouseFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: Literal["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


@app.get("/")
def home():
    return {"message": "Housing Price Prediction API is running"}


@app.post("/predict")
def predict(data: HouseFeatures):
    df = pd.get_dummies(pd.DataFrame([data.model_dump()]), columns=["ocean_proximity"])
    df = df.reindex(columns=FEATURES, fill_value=0).astype(float)
    price = model.predict(scaler.transform(df))[0][0]
    return {"predicted_house_price": float(price)}