"""
baselines.py
------------
Establish the "bar to beat" before any AI.

We reduce Central India to ONE number per day (the region-average rainfall),
then test two dumb predictors and measure their error on a held-out test set:

  1. PERSISTENCE  -> tomorrow = today
  2. CLIMATOLOGY  -> tomorrow = the historical average for that day-of-year

The AI we build next must beat these RMSE/MAE numbers to be worth anything.

Usage:
  python scripts/baselines.py
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

PROC = os.path.join("data", "processed", "central_rain_2005_2023.nc")
FIG_DIR = "outputs"


TEST_START = "2021-01-01"   # 2021-2023 = unseen test years; 2005-2020 = history


def rmse(pred, true):
    return float(np.sqrt(np.nanmean((pred - true) ** 2)))

def mae(pred, true):
    return float(np.nanmean(np.abs(pred - true)))


def main():
    ds = xr.open_dataset(PROC)
    # region-average rainfall: average across lat & lon -> a daily time series
    series = ds["rain"].mean(dim=["lat", "lon"]).to_series()
    series.index = pd.to_datetime(series.index)
    print(f"Daily region-average series: {len(series)} days "
          f"({series.index.min().date()} to {series.index.max().date()})")
    print(f"Overall mean rainfall: {series.mean():.2f} mm/day\n")


    train = series[series.index < TEST_START]
    test = series[series.index >= TEST_START]
    print(f"Train: {len(train)} days (2005-2020)   |   Test: {len(test)} days (2021-2023)\n")


    # prediction for day t = actual value of day t-1
    persist_pred = test.shift(1)          # yesterday's value, aligned to today
    # the first test day has no "yesterday" inside the test set -> borrow last train day
    persist_pred.iloc[0] = train.iloc[-1]

  
    # build the day-of-year average FROM TRAINING DATA ONLY
    doy_avg = train.groupby(train.index.dayofyear).mean()
    clim_pred = pd.Series(
        [doy_avg.get(d, train.mean()) for d in test.index.dayofyear],
        index=test.index
    )


    results = {
        "Persistence (= yesterday)": (rmse(persist_pred.values, test.values),
                                      mae(persist_pred.values, test.values)),
        "Climatology (= seasonal normal)": (rmse(clim_pred.values, test.values),
                                            mae(clim_pred.values, test.values)),
    }

    print("=" * 52)
    print(f"{'BASELINE':<34}{'RMSE':>8}{'MAE':>9}")
    print("-" * 52)
    for name, (r, m) in results.items():
        print(f"{name:<34}{r:>7.2f} {m:>8.2f}")
    print("=" * 52)
    best = min(results.values(), key=lambda x: x[0])[0]
    print(f"\nBest baseline RMSE = {best:.2f} mm/day  <-- this is the bar the AI must beat.\n")

  
    os.makedirs(FIG_DIR, exist_ok=True)

    # (a) bar chart of errors
    plt.figure(figsize=(6, 4))
    names = list(results.keys())
    rmses = [results[n][0] for n in names]
    plt.bar(range(len(names)), rmses, color=["#7C9CBF", "#5BAE9C"])
    plt.xticks(range(len(names)), ["Persistence", "Climatology"])
    plt.ylabel("RMSE (mm/day)  — lower is better")
    plt.title("Baseline error — the bar to beat")
    for i, v in enumerate(rmses):
        plt.text(i, v + 0.05, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "baseline_rmse.png"), dpi=130)
    plt.close()

    
    window = test.iloc[:120]
    plt.figure(figsize=(11, 4))
    plt.plot(window.index, window.values, label="Actual", color="#222", linewidth=1.6)
    plt.plot(window.index, persist_pred.loc[window.index].values,
             label="Persistence", color="#C77", alpha=.8)
    plt.plot(window.index, clim_pred.loc[window.index].values,
             label="Climatology", color="#5BAE9C", alpha=.9)
    plt.ylabel("Rainfall (mm/day)")
    plt.title("Baselines vs reality — first 120 test days (2021)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "baseline_timeseries.png"), dpi=130)
    plt.close()

    print(f"Charts saved: {FIG_DIR}/baseline_rmse.png , {FIG_DIR}/baseline_timeseries.png")


if __name__ == "__main__":
    main()