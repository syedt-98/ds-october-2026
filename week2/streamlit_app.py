import streamlit as st
import requests



# 1. Page Title
st.title("🏠 California Housing Price Predictor")

# 2. Input Form
with st.form("form"):
    longitude = st.number_input("Longitude")
    latitude = st.number_input("Latitude")
    housing_median_age = st.number_input("Housing Age")
    total_rooms = st.number_input("Total Rooms")
    total_bedrooms = st.number_input("Total Bedrooms")
    population = st.number_input("Population")
    households = st.number_input("Households")
    median_income = st.number_input("Median Income")
    ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])
    
    submitted = st.form_submit_button("Predict Price")

# 3. Handle Prediction Request
if submitted:
    payload = {
        "longitude": longitude, "latitude": latitude,
        "housing_median_age": housing_median_age, "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms, "population": population,
        "households": households, "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }
    
    try:
        # Send data to FastAPI backend
        res = requests.post("http://127.0.0.1:8000/predict",json=payload)
        
        if res.status_code == 200:
            price = res.json()["predicted_house_price"]
            st.success(f"### Estimated Price: ${price:,.2f}")
        else:
            st.error(f"API Error: {res.text}")
            
    except requests.exceptions.ConnectionError:
            st.error("FastAPI server is not running! Start it using uvicorn.")