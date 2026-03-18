import streamlit as st
import json, pickle, numpy as np, pandas as pd, requests, plotly.graph_objects as go
from tensorflow import keras
from datetime import datetime, timedelta

st.set_page_config(page_title="Weather Forecast — TVM", page_icon="🌤", layout="wide")


@st.cache_resource
def load_model(region):
    return keras.models.load_model(f"models/{region}_model.keras")


@st.cache_resource
def load_scaler(region):
    with open(f"models/{region}_scaler.pkl", "rb") as f:
        return pickle.load(f)


def fetch_recent(lat, lon, days=3):
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "timezone": "Asia/Kolkata",
            "past_days": days,
            "forecast_days": 1,
        },
    )
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def predict_next24(region, lat, lon):
    model = load_model(region)
    scaler = load_scaler(region)
    df = fetch_recent(lat, lon, days=3).dropna()
    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek
    FEATURES = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "hour",
        "dayofweek",
    ]
    scaled = scaler.transform(df[FEATURES].tail(48))
    X = scaled[-48:].reshape(1, 48, len(FEATURES))
    pred_scaled = model.predict(X, verbose=0)[0]
    dummy = np.zeros((24, scaler.n_features_in_))
    dummy[:, 0] = pred_scaled
    return scaler.inverse_transform(dummy)[:, 0], df


try:
    v = json.load(open("version.json"))
    st.caption(
        f"Model v{v['version']} | Trained: {v['trained_on']} | RMSE Technopark: {v['rmse_technopark']:.2f}°C | RMSE Thampanoor: {v['rmse_thampanoor']:.2f}°C"
    )
except:
    st.caption("version.json not found")

st.title("🌤 Weather Forecast — Thiruvananthapuram")

LOCATIONS = {
    "Technopark": ("technopark", 8.5574, 76.8800),
    "Thampanoor": ("thampanoor", 8.4875, 76.9525),
}

tab1, tab2 = st.tabs(["Technopark", "Thampanoor"])

for tab, (name, (region, lat, lon)) in zip([tab1, tab2], LOCATIONS.items()):
    with tab:
        with st.spinner(f"Forecasting {name}..."):
            forecast, actuals = predict_next24(region, lat, lon)
        future_times = [datetime.now() + timedelta(hours=i) for i in range(24)]
        actual_times = actuals["time"].tolist()[-48:]
        actual_temps = actuals["temperature_2m"].tolist()[-48:]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=actual_times,
                y=actual_temps,
                name="Observed (last 48h)",
                line=dict(color="#4A90D9"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_times,
                y=forecast,
                name="Forecast (next 24h)",
                line=dict(color="#E8593C", dash="dash"),
            )
        )
        fig.update_layout(
            title=f"{name} — Temperature Forecast",
            yaxis_title="Temperature (°C)",
            xaxis_title="Time",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Min forecast", f"{min(forecast):.1f}°C")
        col2.metric("Max forecast", f"{max(forecast):.1f}°C")
        col3.metric("Avg forecast", f"{np.mean(forecast):.1f}°C")
        with st.sidebar:
            st.subheader(f"{name} quick stats")
            recent = actuals.tail(1)
            st.metric("Current temp", f"{recent['temperature_2m'].values[0]:.1f}°C")
            st.metric("Precipitation", f"{recent['precipitation'].values[0]:.1f} mm")
            st.metric("Wind speed", f"{recent['wind_speed_10m'].values[0]:.1f} km/h")
