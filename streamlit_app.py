
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("catboost_model.pkl")

# App Title
st.title("📊 Perth Household Power Consumption Predictor")
st.markdown("Predict electricity consumption based on weather and time features.")

# Sidebar input function
def user_input_features():
    st.sidebar.header("User Input Parameters")

    met_temp = st.sidebar.slider('Temperature (°C)', 0.0, 45.0, 25.0)
    dwpt = st.sidebar.slider('Dew Point (°C)', 0.0, 30.0, 10.0)
    met_humidity = st.sidebar.slider('Humidity (%)', 0.0, 100.0, 50.0)
    met_precip = st.sidebar.slider('Precipitation (mm)', 0.0, 20.0, 0.0)
    met_wind_speed = st.sidebar.slider('Wind Speed (km/h)', 0.0, 100.0, 15.0)
    met_pressure = st.sidebar.slider('Pressure (hPa)', 980.0, 1050.0, 1015.0)

    hour = st.sidebar.slider('Hour of Day', 0, 23, 12)
    dayofweek = st.sidebar.slider('Day of Week (0=Mon)', 0, 6, 2)
    month = st.sidebar.slider('Month (1-12)', 1, 12, 6)
    is_weekend = st.sidebar.selectbox('Is Weekend?', [0, 1])
    is_daylight = st.sidebar.selectbox('Is Daylight?', [0, 1])
    is_public_holiday = st.sidebar.selectbox('Is Public Holiday?', [0, 1])
    solar_elevation = st.sidebar.slider('Solar Elevation (°)', -90.0, 90.0, 45.0)

    season = st.sidebar.selectbox('Season', ['Summer', 'Winter', 'Spring'])
    season_Summer = int(season == 'Summer')
    season_Winter = int(season == 'Winter')
    season_Spring = int(season == 'Spring')

    data = {
        'met_temp': met_temp,
        'dwpt': dwpt,
        'met_humidity': met_humidity,
        'met_precip': met_precip,
        'met_wind_speed': met_wind_speed,
        'met_pressure': met_pressure,
        'hour': hour,
        'dayofweek': dayofweek,
        'month': month,
        'is_weekend': is_weekend,
        'is_daylight': is_daylight,
        'is_public_holiday': is_public_holiday,
        'solar_elevation': solar_elevation,
        'season_Summer': season_Summer,
        'season_Winter': season_Winter,
        'season_Spring': season_Spring
    }

    return pd.DataFrame([data])

# Load inputs
input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)[0]

# Output
st.subheader("📈 Predicted Electricity Consumption (kWh)")
st.metric(label="Consumption", value=f"{prediction:.3f} kWh")

st.markdown("---")
st.caption("Developed for ML-Based Power Consumption Modeling – Curtin University")
