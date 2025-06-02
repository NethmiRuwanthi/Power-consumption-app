import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

# Load the trained CatBoost model
model = joblib.load("catboost_model.pkl")

# Page config
st.set_page_config(page_title="⚡ WA Power Consumption Predictor", layout="wide")

# Sidebar with prediction summary
st.sidebar.header("📊 Prediction Summary")
sidebar_placeholder = st.sidebar.empty()

# Main title
st.title("⚡ WA Household Power Consumption Predictor")
st.markdown("Use the form below to input conditions and forecast your household electricity consumption.")

# Input fields
with st.form("prediction_form"):
    generation_kWh = st.number_input("Solar Generation (kWh)", 0.0, 10.0, 0.5)
    met_precip = st.number_input("Precipitation", 0.0, 100.0, 0.0)
    met_wind_dir = st.number_input("Wind Direction", 0.0, 360.0, 180.0)
    met_wind_speed = st.number_input("Wind Speed", 0.0, 100.0, 10.0)
    met_pressure = st.number_input("Pressure (hPa)", 950.0, 1050.0, 1015.0)
    coco = st.selectbox("Cloud Cover (code)", [0, 1, 2, 3])
    hour = st.slider("Hour of Day", 0, 23, 12)
    dayofweek = st.slider("Day of Week (0=Mon)", 0, 6, 2)
    month = st.slider("Month", 1, 12, 6)
    is_weekend = st.selectbox("Is Weekend", [0, 1])
    lag_1 = st.slider("Lag 1", 0.0, 2.0, 0.35)
    lag_2 = st.slider("Lag 2", 0.0, 2.0, 0.32)
    lag_48 = st.slider("Lag 48", 0.0, 2.0, 0.45)
    roll_mean_6 = st.slider("Rolling Mean (6)", 0.0, 2.0, 0.34)
    roll_std_6 = st.slider("Rolling Std (6)", 0.0, 1.0, 0.02)
    roll_mean_12 = st.slider("Rolling Mean (12)", 0.0, 2.0, 0.36)
    roll_std_12 = st.slider("Rolling Std (12)", 0.0, 1.0, 0.03)
    roll_mean_48 = st.slider("Rolling Mean (48)", 0.0, 2.0, 0.37)
    roll_std_48 = st.slider("Rolling Std (48)", 0.0, 1.0, 0.025)
    is_daylight = st.selectbox("Is Daylight", [0, 1])
    tariff_period = st.slider("Tariff Period", 1, 3, 2)
    roll_mean_336 = st.slider("Rolling Mean (336)", 0.0, 2.0, 0.38)
    THI = st.slider("THI", 0.0, 50.0, 21.0)
    is_peak_hour = st.selectbox("Is Peak Hour", [0, 1])
    season_Spring = st.selectbox("Season: Spring", [0, 1])
    season_Summer = st.selectbox("Season: Summer", [0, 1])
    season_Winter = st.selectbox("Season: Winter", [0, 1])
    submitted = st.form_submit_button("🔮 Predict")

# Prediction
if submitted:
    input_df = pd.DataFrame([[
        generation_kWh, met_precip, met_wind_dir, met_wind_speed, met_pressure, coco, hour,
        dayofweek, month, is_weekend, lag_1, lag_2, lag_48, roll_mean_6, roll_std_6,
        roll_mean_12, roll_std_12, roll_mean_48, roll_std_48, is_daylight, tariff_period,
        roll_mean_336, THI, is_peak_hour, season_Spring, season_Summer, season_Winter
    ]], columns=[
        'generation_kWh', 'met_precip', 'met_wind_dir', 'met_wind_speed', 'met_pressure',
        'coco', 'hour', 'dayofweek', 'month', 'is_weekend', 'lag_1', 'lag_2', 'lag_48',
        'roll_mean_6', 'roll_std_6', 'roll_mean_12', 'roll_std_12', 'roll_mean_48',
        'roll_std_48', 'is_daylight', 'tariff_period', 'roll_mean_336', 'THI',
        'is_peak_hour', 'season_Spring', 'season_Summer', 'season_Winter'
    ])

    prediction = model.predict(input_df)[0]
    estimated_cost = prediction * 0.32  # assumed rate in AUD/kWh

    # Show sidebar summary
    with st.sidebar:
        st.metric("🔌 Consumption (kWh)", f"{prediction:.3f}")
        st.metric("💰 Estimated Cost", f"${estimated_cost:.2f}")

    # Feedback message
    if prediction < 0.25:
        st.success("🌿 Excellent! Low usage detected.")
    elif prediction < 0.5:
        st.info("⚖️ Moderate usage.")
    else:
        st.warning("🔥 High usage! Consider reducing consumption.")

    # SHAP local explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)
    st.subheader("🔍 Feature Impact (SHAP)")
    shap.plots.waterfall(shap_values[0], max_display=10)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(bbox_inches="tight")

    # Top influencing features
    st.subheader("📌 Top Influencing Features")
    top_idx = np.argsort(np.abs(shap_values[0].values))[-3:][::-1]
    for i in top_idx:
        st.write(f"- **{input_df.columns[i]}**: SHAP = {shap_values[0].values[i]:.4f}")

    # Radar chart of input profile
    st.subheader("📊 Input Feature Profile")
    radar_data = input_df.copy()
    radar_data = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())
    radar_data = pd.concat([radar_data.T, radar_data.T.iloc[:1]])  # to close the loop

    fig = px.line_polar(radar_data, r=radar_data[0], theta=radar_data.index, line_close=True)
    st.plotly_chart(fig)

    # Notes box
    st.subheader("📝 Your Notes")
    st.text_area("Type any thoughts or follow-up actions here...", height=150)

    # Footer
    st.markdown("---")
    st.markdown("<small>🔧 Built with 💚 using Streamlit • Project by Nethmi Ruwanthi</small>", unsafe_allow_html=True)
