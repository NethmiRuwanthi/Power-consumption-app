import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from datetime import datetime

# ✅ Set page config
st.set_page_config(page_title="Perth Power Consumption Predictor", layout="centered")

# ✅ Custom Styling (Light theme only)
st.markdown("""
<style>
    html, body, .main, .block-container {
        background-color: #f9fcff;
        font-family: 'Segoe UI';
        padding: 1rem;
    }
    h1, h2, h3, .stMarkdown h3 { color: #1a66ff; }
    .stButton>button { background-color: #1a66ff; color: white; border-radius: 10px; }
    .stDownloadButton > button { background-color: #33a673; color: white; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ✅ Sidebar Inputs
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
        'lag_1': 0.3,
        'lag_2': 0.3,
        'lag_48': 0.3,
        'roll_mean_6': 0.3,
        'roll_std_6': 0.1,
        'roll_mean_12': 0.3,
        'roll_std_12': 0.1,
        'roll_mean_48': 0.3,
        'roll_std_48': 0.1,
        'is_daylight': 1,
        'tariff_period': 1,
        'roll_mean_336': 0.3,
        'THI': 20.0,
        'is_peak_hour': 0,
        'season_Spring': season_Spring,
        'season_Summer': season_Summer,
        'season_Winter': season_Winter
    }
    return pd.DataFrame([data])

# ✅ Load Model
model = joblib.load("catboost_model.pkl")

# ✅ App Title
st.title("🔋 Perth Household Power Consumption Predictor")
st.markdown("##### _Predict consumption using AI-powered insights – by SK Nethmi Ruwanthi_")

# ✅ About Section
with st.expander("📘 About this App"):
    st.write("""
    This app predicts household energy consumption in Perth using a tuned CatBoost machine learning model.
    It considers weather data, time-based features, and engineered lags to make accurate predictions.
    Built for the Curtin University Predictive Analytics Project 2025.
    """)

# ✅ Run Prediction
input_df = user_input_features()
expected_cols = list(input_df.columns)
input_df = input_df.reindex(columns=expected_cols, fill_value=0)
prediction = model.predict(input_df)[0]

st.markdown("## 🔎 Estimate Energy Use Based on Your Inputs")
st.success(f"Estimated consumption: {prediction:.3f} kWh")

if st.checkbox("📉 Compare to Daily Average"):
    plt.figure(figsize=(5, 3))
    plt.bar(["Your Prediction", "Typical Avg"], [prediction, 0.6], color=["blue", "gray"])
    plt.ylabel("kWh")
    st.pyplot(plt)

if st.button("📅 Download prediction as CSV"):
    output = input_df.copy()
    output["predicted_kWh"] = prediction
    st.download_button("Download", output.to_csv(index=False), file_name="prediction.csv", mime="text/csv")

# ✅ Prophet Forecasting
st.markdown("## Forecast Energy Use for Future Dates")

@st.cache_data
def load_time_series():
    date_rng = pd.date_range(start='2024-01-01', periods=180, freq='D')
    np.random.seed(42)
    consumption = 0.6 + np.sin(np.linspace(0, 3 * np.pi, len(date_rng))) * 0.1 + np.random.normal(0, 0.05, len(date_rng))
    return pd.DataFrame({'ds': date_rng, 'y': consumption})

ts_data = load_time_series()
m = Prophet()
m.fit(ts_data)

selected_date = st.date_input("Forecast Date", value=pd.to_datetime("2025-06-15"))
last_date = ts_data['ds'].max().date()
selected_date_dt = pd.to_datetime(selected_date).date()
days_ahead = (selected_date_dt - last_date).days
forecast_horizon = max(60, days_ahead + 1)

future = m.make_future_dataframe(periods=forecast_horizon)
forecast = m.predict(future)
selected_forecast = forecast[forecast['ds'] == pd.to_datetime(selected_date)]

if not selected_forecast.empty:
    yhat = selected_forecast['yhat'].values[0]
    st.success(f"📆 Forecasted consumption on **{selected_date}**: **{yhat:.3f} kWh**")
else:
    st.warning("⚠️ Selected date is outside forecast range. Please select a closer date.")

if st.checkbox("📊 Show Forecast Chart"):
    st.markdown("### 📈 Daily Forecast Trend")
    fig = plot_plotly(m, forecast)
    fig.update_layout(xaxis_title="Date", yaxis_title="Predicted kWh", title="Energy Forecast (Prophet)")
    st.plotly_chart(fig, use_container_width=True)

if st.checkbox("📋 Show Forecast Table"):
    st.markdown("### 📋 Forecast Table (Next 60 Days)")
    styled_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(60)
    styled_table.columns = ['Date', 'Predicted kWh', 'Lower Bound', 'Upper Bound']
    st.dataframe(styled_table.style.format({"Predicted kWh": "{:.2f}", "Lower Bound": "{:.2f}", "Upper Bound": "{:.2f}"}))

st.markdown("---")
st.markdown("🛠️ *Developed for the Curtin University Predictive Analytics Project – 2025*")
st.caption("By SK Nethmi Ruwanthi – Powered with CatBoost, Prophet & Streamlit")
