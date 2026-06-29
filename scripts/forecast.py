"""
forecast.py
-----------
Turn the trained 1-day LSTM into a multi-day forecast WITH an uncertainty band.

Two ideas:
  1. RECURSIVE forecast - predict day+1, feed it back to predict day+2, etc.
  2. UNCERTAINTY - measure the model's real error at each horizon across the
     test set; that measured spread becomes the confidence band (it widens
     further out, because recursive errors compound).

Also saves outputs/forecast.json  (the feed the dashboard will read).

Run:  python scripts/forecast.py
"""

import os, json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import torch

# reuse the model definition + settings from model.py
from model import RainLSTM, WINDOW, TEST_START, PROC

FIG_DIR = "outputs"
MODEL_PATH = os.path.join("models", "lstm_rain.pt")
HORIZON = 7          # forecast 7 days ahead
Z = 1.96             # 95% band (1.96 standard deviations)


def forecast_ahead(model, history_norm, horizon):
    
    seq = list(history_norm)
    preds = []
    for _ in range(horizon):
        x = torch.tensor(np.array(seq[-WINDOW:], dtype="float32")[None, :, None])
        with torch.no_grad():
            p = model(x).item()
        preds.append(p)
        seq.append(p)            # feed the prediction back in
    return np.array(preds)       # still normalised


def main():
    
    ds = xr.open_dataset(PROC)
    series = ds["rain"].mean(dim=["lat", "lon"]).to_series()
    series.index = pd.to_datetime(series.index)
    values = series.values.astype("float32")
    dates = series.index
    mu = values[dates < TEST_START].mean()
    sd = values[dates < TEST_START].std()
    norm = (values - mu) / sd

   
    model = RainLSTM()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    
    test_idx = np.where(dates >= np.datetime64(TEST_START))[0]
    starts = [i for i in test_idx if i >= WINDOW and i + HORIZON <= len(values)]
    errors = [[] for _ in range(HORIZON)]
    for s in starts[::2]:                       # every 2nd start = plenty, faster
        fc = forecast_ahead(model, norm[s - WINDOW:s], HORIZON)
        actual = norm[s:s + HORIZON]
        for h in range(HORIZON):
            errors[h].append(fc[h] - actual[h])
    band_mm = np.array([np.std(e) for e in errors]) * sd   # spread in mm, per horizon
    print("\nUncertainty (1 std, mm) by lead day:")
    for h in range(HORIZON):
        print(f"  day +{h+1}: +/- {band_mm[h]:.2f} mm")

    
    last_hist = norm[-WINDOW:]
    fc_norm = forecast_ahead(model, last_hist, HORIZON)
    fc_mm = np.clip(fc_norm * sd + mu, 0, None)
    lower = np.clip(fc_mm - Z * band_mm, 0, None)
    upper = fc_mm + Z * band_mm
    fc_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=HORIZON)

    print(f"\n{HORIZON}-day forecast from {dates[-1].date()}:")
    for d, v, lo, hi in zip(fc_dates, fc_mm, lower, upper):
        print(f"  {d.date()}  {v:5.2f} mm   (95% range {lo:.2f} - {hi:.2f})")

    
    os.makedirs(FIG_DIR, exist_ok=True)
    payload = {
        "region": "Central India",
        "issued_from": str(dates[-1].date()),
        "unit": "mm/day",
        "forecast": [
            {"date": str(d.date()), "rain": round(float(v), 2),
             "low": round(float(lo), 2), "high": round(float(hi), 2)}
            for d, v, lo, hi in zip(fc_dates, fc_mm, lower, upper)
        ],
    }
    with open(os.path.join(FIG_DIR, "forecast.json"), "w") as f:
        json.dump(payload, f, indent=2)

   
    tail = 30
    plt.figure(figsize=(11, 4.5))
    plt.plot(dates[-tail:], values[-tail:], color="#222", linewidth=1.6, label="Recent actual")
    plt.plot(fc_dates, fc_mm, color="#2C8C99", linewidth=2, marker="o", label="Forecast")
    plt.fill_between(fc_dates, lower, upper, color="#2C8C99", alpha=.20, label="95% uncertainty")
    plt.axvline(dates[-1], color="#999", linestyle="--", linewidth=1)
    plt.text(dates[-1], plt.ylim()[1]*0.9, " forecast starts", color="#666", fontsize=9)
    plt.ylabel("Rainfall (mm/day)")
    plt.title(f"Central India - {HORIZON}-day rainfall forecast with uncertainty")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "forecast.png"), dpi=130)
    plt.close()
    print(f"\nSaved: {FIG_DIR}/forecast.png , {FIG_DIR}/forecast.json")


if __name__ == "__main__":
    main()