from pathlib import Path

# Define the updated Streamlit app code with CSS styling included
updated_code = """
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import datetime

# Load model
model = joblib.load("catboost_model.pkl")

st.set_page_config(page_title="Perth Power Consumption Predictor", layout="centered")
st.markdown(\"""
<style>
    .css-1d391kg {color: #1a66ff;}
    .css-18e3th9 {background-color: #f5faff;}
</style>
\""", unsafe_allow_html=True)

st.title("🔋 Perth Household Power Consumption Predictor")
st.markdown("Predict electricity consumption based on weather, time, and engineered features. Author: SK Nethmi Ruwanthi")

# About section
with st.expander("📘 About this App"):
    st.write(\"""
    This app predicts household energy consumption in Perth using a tuned CatBoost machine learning model.
    It considers weather data, time-based features, and engineered lags to make accurate predictions.
    Built for the Curtin University Predictive Analytics Project 2025.
    \""")

# Current time display
now = datetime.datetime.now()
st.sidebar.markdown(f"🕒 **Current Time:** {now.strftime('%A, %d %B %Y, %H:%M')}")

# Sidebar inputs
st.sidebar.header("Input Features")

def user_input_features():
    generation_kWh = st.sidebar.slider('Solar Generation (kWh)', 0.0, 5.0, 0.0)
    met_precip = st.sidebar.slider('Precipitation (mm)', 0.0, 20.0, 0.0)
    met_wind_dir = st.sidebar.slider('Wind Direction (°)', 0.0, 360.0, 180.0)
    met_wind_speed = st.sidebar.slider('Wind Speed (km/h)', 0.0, 100.0, 10.0)
    met_pressure = st.sidebar.slider('Pressure (hPa)', 980.0, 1050.0, 1015.0)
    coco = st.sidebar.slider('Cloud Cover Index', 0.0, 9.0, 1.0)
    hour = st.sidebar.slider('Hour of Day', 0, 23, 12)
    dayofweek = st.sidebar.slider('Day of Week (0=Mon)', 0, 6, 3)
    month = st.sidebar.slider('Month (1-12)', 1, 12, 6)
    is_weekend = st.sidebar.selectbox('Is Weekend?', [0, 1])
    lag_1 = st.sidebar.number_input('Lag 1 (kWh)', 0.0, 5.0, 0.3)
    lag_2 = st.sidebar.number_input('Lag 2 (kWh)', 0.0, 5.0, 0.3)
    lag_48 = st.sidebar.number_input('Lag 48 (kWh)', 0.0, 5.0, 0.3)
    roll_mean_6 = st.sidebar.number_input('Rolling Mean (6)', 0.0, 5.0, 0.3)
    roll_std_6 = st.sidebar.number_input('Rolling Std (6)', 0.0, 2.0, 0.1)
    roll_mean_12 = st.sidebar.number_input('Rolling Mean (12)', 0.0, 5.0, 0.3)
    roll_std_12 = st.sidebar.number_input('Rolling Std (12)', 0.0, 2.0, 0.1)
    roll_mean_48 = st.sidebar.number_input('Rolling Mean (48)', 0.0, 5.0, 0.3)
    roll_std_48 = st.sidebar.number_input('Rolling Std (48)', 0.0, 2.0, 0.1)
    is_daylight = st.sidebar.selectbox('Is Daylight?', [0, 1])
    tariff_period = st.sidebar.slider('Tariff Period (0–3)', 0, 3, 1)
    roll_mean_336 = st.sidebar.number_input('Rolling Mean (336)', 0.0, 5.0, 0.3)
    THI = st.sidebar.number_input('Temperature Humidity Index (THI)', 0.0, 40.0, 20.0)
    is_peak_hour = st.sidebar.selectbox('Is Peak Hour?', [0, 1])
    season = st.sidebar.selectbox('Season', ['Summer', 'Winter', 'Spring'])
    season_Summer = int(season == 'Summer')
    season_Winter = int(season == 'Winter')
    season_Spring = int(season == 'Spring')

    data = {
        'generation_kWh': generation_kWh,
        'met_precip': met_precip,
        'met_wind_dir': met_wind_dir,
        'met_wind_speed': met_wind_speed,
        'met_pressure': met_pressure,
        'coco': coco,
        'hour': hour,
        'dayofweek': dayofweek,
        'month': month,
        'is_weekend': is_weekend,
        'lag_1': lag_1,
        'lag_2': lag_2,
        'lag_48': lag_48,
        'roll_mean_6': roll_mean_6,
        'roll_std_6': roll_std_6,
        'roll_mean_12': roll_mean_12,
        'roll_std_12': roll_std_12,
        'roll_mean_48': roll_mean_48,
        'roll_std_48': roll_std_48,
        'is_daylight': is_daylight,
        'tariff_period': tariff_period,
        'roll_mean_336': roll_mean_336,
        'THI': THI,
        'is_peak_hour': is_peak_hour,
        'season_Spring': season_Spring,
        'season_Summer': season_Summer,
        'season_Winter': season_Winter
    }

    return pd.DataFrame([data])

input_df = user_input_features()

expected_cols = ['generation_kWh', 'met_precip', 'met_wind_dir', 'met_wind_speed', 'met_pressure', 'coco',
                 'hour', 'dayofweek', 'month', 'is_weekend', 'lag_1', 'lag_2', 'lag_48', 'roll_mean_6',
                 'roll_std_6', 'roll_mean_12', 'roll_std_12', 'roll_mean_48', 'roll_std_48', 'is_daylight',
                 'tariff_period', 'roll_mean_336', 'THI', 'is_peak_hour', 'season_Spring', 'season_Summer',
                 'season_Winter']
input_df = input_df.reindex(columns=expected_cols, fill_value=0)

prediction = model.predict(input_df)[0]

st.subheader("📈 Predicted Energy Consumption")
st.success(f"Estimated consumption: {prediction:.3f} kWh")

if st.checkbox("📉 Compare to Daily Average"):
    plt.figure(figsize=(5, 3))
    plt.bar(["Your Prediction", "Typical Avg"], [prediction, 0.6], color=["blue", "gray"])
    plt.ylabel("kWh")
    st.pyplot(plt)

if st.button("📥 Download prediction as CSV"):
    output = input_df.copy()
    output["predicted_kWh"] = prediction
    st.download_button("Download", output.to_csv(index=False), file_name="prediction.csv", mime="text/csv")

st.markdown("---")
st.caption("Developed for ML-Based Power Consumption Modeling – Curtin University")
"""

# Save to file
file_path = Path("/mnt/data/streamlit_app_styled.py")
file_path.write_text(updated_code)

file_path.name

