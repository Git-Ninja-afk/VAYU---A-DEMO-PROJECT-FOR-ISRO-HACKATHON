"""
warm_nights.py  --  hot-night heat stress from IMD min-temperature data
-----------------------------------------------------------------------
Mirror of heatwave.py, but for NIGHTS. A "warm night" (tropical night) is a
night the air never cools below a threshold, so the body can't recover --
a recognised heat-health hazard, especially during heatwaves.

What it does:
  1. LOAD   - open the processed tmin file made by preprocess.py
  2. COUNT  - nights/year where Tmin >= 25 C  (tropical-night threshold)
  3. WARM   - shift the field by +0.5, +1, +2 ... C and recount
  4. SAVE   - outputs/warm_nights.json

Threshold note:
  25 C is the common "tropical night" / warm-night heat-health threshold.
  (IMD also has an official warm-night definition tied to a hot day plus a
  Tmin departure from normal; the absolute 25 C cut is simpler to defend and
  to read on a dashboard.)

Run (after `python3 scripts/preprocess.py --vars tmin`):
  python3 scripts/warm_nights.py
"""

import os
import json
import numpy as np
import xarray as xr

PROC_DIR = os.path.join("data", "processed")
OUT = "outputs"

THRESHOLD_C = 25.0                       # tropical-night threshold (Tmin)
WARMING_STEPS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def find_tmin_file():
    if not os.path.isdir(PROC_DIR):
        raise FileNotFoundError("Run: python3 scripts/preprocess.py --vars tmin")
    cands = [f for f in os.listdir(PROC_DIR)
             if f.startswith("central_tmin") and f.endswith(".nc")]
    if not cands:
        raise FileNotFoundError(
            f"No central_tmin_*.nc in {PROC_DIR}/. "
            "Run: python3 scripts/preprocess.py --vars tmin"
        )
    return os.path.join(PROC_DIR, sorted(cands)[-1])


def warm_nights(tmin, valid, cell_is_land, delta=0.0):
    """Average warm nights per year at a typical land cell, after +delta C."""
    hot = ((tmin + delta) >= THRESHOLD_C) & valid
    per_year = hot.groupby("time.year").sum("time")
    per_year = per_year.where(cell_is_land)
    return float(per_year.mean(["lat", "lon"]).mean("year"))


def main():
    path = find_tmin_file()
    print(f"Reading {path}")
    ds = xr.open_dataset(path)
    tmin = ds["tmin"]

    valid = tmin.notnull()
    cell_is_land = valid.mean("time") > 0.5

    n_years = len(np.unique(tmin["time"].dt.year))
    n_land = int(cell_is_land.sum())
    print(f"  years: {n_years} | land cells in box: {n_land} | threshold: {THRESHOLD_C} C\n")

    response = {d: round(warm_nights(tmin, valid, cell_is_land, d), 1)
                for d in WARMING_STEPS}
    baseline = response[0.0]
    per_degC_local = round(response[1.0] - response[0.0], 1)

    print("=== Warm nights / year (Tmin >= 25 C), from IMD data ===")
    print(f"  baseline (today)        : {baseline} nights/yr")
    for d in WARMING_STEPS[1:]:
        print(f"  with +{d:.1f} C warming    : {response[d]} nights/yr  "
              f"({response[d]-baseline:+.1f})")
    print(f"\n  local sensitivity       : ~{per_degC_local} extra nights per +1 C")

    os.makedirs(OUT, exist_ok=True)
    payload = {
        "region": "Central India",
        "threshold_c": THRESHOLD_C,
        "definition": "Tropical night: daily Tmin >= 25 C",
        "source": os.path.basename(path),
        "baseline_warm_nights": baseline,
        "warming_response": {str(d): response[d] for d in WARMING_STEPS},
        "nights_per_degC_near_baseline": per_degC_local,
    }
    with open(os.path.join(OUT, "warm_nights.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {OUT}/warm_nights.json")


if __name__ == "__main__":
    main()