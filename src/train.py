import numpy as np, json, yaml, pickle, subprocess
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import math, joblib

p = yaml.safe_load(open("params.yaml"))["train"]
metrics_out = {}


def rmse(y_true, y_pred):
    return math.sqrt(np.mean((y_true - y_pred) ** 2))


for region in ["technopark", "thampanoor"]:
    X_train = np.load(f"data/processed/{region}_X_train.npy")
    y_train = np.load(f"data/processed/{region}_y_train.npy")
    X_test = np.load(f"data/processed/{region}_X_test.npy")
    y_test = np.load(f"data/processed/{region}_y_test.npy")

    # Flatten 3D to 2D for sklearn
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)

    model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100))
    model.fit(X_train_2d, y_train)
    pred = model.predict(X_test_2d)

    with open(f"models/{region}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    dummy = np.zeros((pred.shape[0] * pred.shape[1], scaler.n_features_in_))
    dummy[:, 0] = pred.flatten()
    inv_pred = scaler.inverse_transform(dummy)[:, 0].reshape(pred.shape)
    dummy2 = np.zeros_like(dummy)
    dummy2[:, 0] = y_test.flatten()
    inv_true = scaler.inverse_transform(dummy2)[:, 0].reshape(y_test.shape)

    metrics_out[region] = {
        "mae": float(mean_absolute_error(inv_true.flatten(), inv_pred.flatten())),
        "rmse": float(rmse(inv_true.flatten(), inv_pred.flatten())),
    }
    joblib.dump(model, f"models/{region}_model.pkl")
    print(f"{region} RMSE: {metrics_out[region]['rmse']:.2f}")

json.dump(metrics_out, open("metrics.json", "w"), indent=2)

try:
    sha = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode()
        .strip()
    )
except:
    sha = "unknown"

version = {
    "version": datetime.now().strftime("%Y%m%d"),
    "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "git_sha": sha,
    "rmse_technopark": metrics_out["technopark"]["rmse"],
    "rmse_thampanoor": metrics_out["thampanoor"]["rmse"],
    "mae_technopark": metrics_out["technopark"]["mae"],
    "mae_thampanoor": metrics_out["thampanoor"]["mae"],
}
json.dump(version, open("version.json", "w"), indent=2)
print("version.json written:", version)
