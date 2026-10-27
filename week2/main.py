# from typing import Literal

# import joblib
# import pandas as pd
# import tensorflow as tf
# from fastapi import FastAPI
# from pydantic import BaseModel

# app = FastAPI(title="Housing Price Prediction API")

# model = tf.keras.models.load_model("best_model.keras")
# scaler = joblib.load("scaler.pkl")

# FEATURES = [
#     "longitude", "latitude", "housing_median_age", "total_rooms",
#     "total_bedrooms", "population", "households", "median_income",
#     "ocean_proximity_<1H OCEAN", "ocean_proximity_INLAND", "ocean_proximity_ISLAND",
#     "ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN",
# ]


# class HouseFeatures(BaseModel):
#     longitude: float
#     latitude: float
#     housing_median_age: float
#     total_rooms: float
#     total_bedrooms: float
#     population: float
#     households: float
#     median_income: float
#     ocean_proximity: Literal["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


# @app.get("/")
# def home():
#     return {"message": "Housing Price Prediction API is running"}


# @app.post("/predict")
# def predict(data: HouseFeatures):
#     df = pd.get_dummies(pd.DataFrame([data.model_dump()]), columns=["ocean_proximity"])
#     df = df.reindex(columns=FEATURES, fill_value=0).astype(float)
#     price = model.predict(scaler.transform(df))[0][0]
#     return {"predicted_house_price": float(price)}




import os
from typing import Literal

import joblib
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_NAME = "housing-price-model"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

app = FastAPI(title="Housing Price Prediction API")

FEATURES = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "ocean_proximity_<1H OCEAN", "ocean_proximity_INLAND", "ocean_proximity_ISLAND",
    "ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN",
]

# Model load: pehle MLflow registry ka champion, nahi mila to local file
model_info = {"source": "local file", "name": "best_model.keras", "version": None, "alias": None}
try:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_URI)
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
    model = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}@champion")
    model_info = {"source": "mlflow registry", "name": MODEL_NAME,
                  "version": mv.version, "alias": "champion"}
except Exception as e:
    print("Registry load failed, using local model:", e)
    model = tf.keras.models.load_model("best_model.keras")

scaler = joblib.load("scaler.pkl")


class HouseFeatures(BaseModel):
    longitude: float = Field(..., ge=-125, le=-114)
    latitude: float = Field(..., ge=32, le=43)
    housing_median_age: float = Field(..., ge=0, le=100)
    total_rooms: float = Field(..., gt=0)
    total_bedrooms: float = Field(..., gt=0)
    population: float = Field(..., gt=0)
    households: float = Field(..., gt=0)
    median_income: float = Field(..., gt=0)
    ocean_proximity: Literal["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def get_model_info():
    return {**model_info, "features": FEATURES, "n_features": len(FEATURES)}


@app.post("/predict")
def predict(data: HouseFeatures):
    try:
        df = pd.get_dummies(pd.DataFrame([data.model_dump()]), columns=["ocean_proximity"])
        df = df.reindex(columns=FEATURES, fill_value=0).astype(float)
        price = model.predict(scaler.transform(df), verbose=0)[0][0]
        return {"predicted_house_price": float(price)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")