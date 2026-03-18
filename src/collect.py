import requests, pandas as pd
from datetime import datetime, timedelta
import yaml, os

params = yaml.safe_load(open("params.yaml"))["collect"]
END = datetime.now().date()
START = END - timedelta(days=params["days_history"])

LOCATIONS = {
    "technopark": (params["technopark_lat"], params["technopark_lon"]),
    "thampanoor": (params["thampanoor_lat"], params["thampanoor_lon"]),
}


def fetch(lat, lon, start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    r = requests.get(
        url,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": str(start),
            "end_date": str(end),
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "timezone": "Asia/Kolkata",
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()["hourly"]
    return pd.DataFrame(data)


for name, (lat, lon) in LOCATIONS.items():
    path = f"data/raw/{name}.csv"
    if os.path.exists(path):
        existing = pd.read_csv(path, parse_dates=["time"])
        last = existing["time"].max().date()
        if last >= END - timedelta(days=1):
            print(f"{name}: up to date")
            continue
        new_df = fetch(lat, lon, last + timedelta(days=1), END)
        new_df["time"] = pd.to_datetime(new_df["time"])
        df = pd.concat([existing, new_df]).drop_duplicates("time")
    else:
        df = fetch(lat, lon, START, END)
        df["time"] = pd.to_datetime(df["time"])
    df.to_csv(path, index=False)
    print(f"{name}: saved {len(df)} rows")
