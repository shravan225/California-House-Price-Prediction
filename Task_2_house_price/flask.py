import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

model = joblib.load('best_model.joblib')
scaler = joblib.load('scaler.joblib')

st.title("California House Price Prediction")

st.header("Input Features")
col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Median Income (MedInc)", min_value=0.0, max_value=15.0, value=3.87)
    house_age = st.number_input("House Age (HouseAge)", min_value=1.0, max_value=52.0, value=28.64)
    ave_rooms = st.number_input("Average Rooms (AveRooms)", min_value=0.0, max_value=141.0, value=5.43)
    ave_bedrms = st.number_input("Average Bedrooms (AveBedrms)", min_value=0.0, max_value=34.0, value=1.10)

with col2:
    population = st.number_input("Population", min_value=3.0, max_value=35682.0, value=1425.48)
    ave_occup = st.number_input("Average Occupancy (AveOccup)", min_value=0.0, max_value=1243.0, value=3.07)
    latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=35.63)
    longitude = st.number_input("Longitude", min_value=-124.0, max_value=-114.0, value=-119.57)

distance_from_coast = np.sqrt((latitude - 34) ** 2 + (longitude + 118) ** 2)
rooms_per_bedroom = ave_rooms / ave_bedrms

input_data = np.array([med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude, distance_from_coast, rooms_per_bedroom]).reshape(1, -1)
scaled_input_data = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(scaled_input_data)
    
    price_usd = prediction[0] * 100000

    exchange_rate = 83
    price_inr = price_usd * exchange_rate

    st.success(f"Predicted House Price:{prediction[0]:,.2f}")
    st.success(f"Predicted House Price (USD): ${price_usd:,.2f}")
    st.success(f"Predicted House Price (INR): ₹{price_inr:,.2f}")

    st.header("Dataset Description")
    st.write("""
    The California Housing dataset contains information on housing prices in California
    The prices are in units of $100,000 (e.g., a value of 2.15  means  $215,000).
    """)
