import pandas as pd, numpy as np, yaml, pickle
from sklearn.preprocessing import MinMaxScaler

p = yaml.safe_load(open("params.yaml"))["preprocess"]
LOOKBACK, HORIZON = p["lookback"], p["horizon"]
FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "hour",
    "dayofweek",
]


def make_windows(df):
    df = df.dropna(subset=["temperature_2m"]).copy()
    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURES])
    X, y = [], []
    for i in range(LOOKBACK, len(scaled) - HORIZON):
        X.append(scaled[i - LOOKBACK : i])
        y.append(scaled[i : i + HORIZON, 0])
    return np.array(X), np.array(y), scaler


for region in ["technopark", "thampanoor"]:
    df = pd.read_csv(f"data/raw/{region}.csv", parse_dates=["time"])
    X, y, scaler = make_windows(df)
    split = int(
        len(X)
        * (1 - float(yaml.safe_load(open("params.yaml"))["preprocess"]["test_split"]))
    )
    np.save(f"data/processed/{region}_X_train.npy", X[:split])
    np.save(f"data/processed/{region}_y_train.npy", y[:split])
    np.save(f"data/processed/{region}_X_test.npy", X[split:])
    np.save(f"data/processed/{region}_y_test.npy", y[split:])
    with open(f"models/{region}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"{region}: X={X.shape}")
