# 🌤 Weather Forecast — Thiruvananthapuram

An MLOps project that forecasts temperature for Technopark and Thampanoor using an LSTM model.

## Features
- Live weather forecast for next 24 hours
- Auto retrains every day via GitHub Actions
- Built with DVC, TensorFlow, and Streamlit

## Locations
- Technopark (8.5574°N, 76.8800°E)
- Thampanoor (8.4875°N, 76.9525°E)

## Tech Stack
- Data: Open-Meteo API
- Model: LSTM (Keras/TensorFlow)
- Pipeline: DVC
- App: Streamlit
- CI/CD: GitHub Actions

## How to Run
```bash
pip install -r requirements.txt
dvc repro
streamlit run app.py
```

## Model Performance
- Technopark RMSE: ~1.07°C
- Thampanoor RMSE: ~0.99°C