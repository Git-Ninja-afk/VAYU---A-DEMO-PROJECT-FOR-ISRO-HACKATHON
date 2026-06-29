"""
heatwave.py  --  turn real IMD max-temperature data into heatwave numbers
-------------------------------------------------------------------------
Replaces the two hard-coded guesses in scenario.py
    BASE_HEATWAVE_DAYS = 18      (guess)
    HEATWAVE_PER_DEGC  = 6       (guess)
with numbers COMPUTED from IMD tmax data for Central India.

What it does:
  1. LOAD   - open the processed tmax file made by preprocess.py
  2. COUNT  - days/year where Tmax >= 40 C  (IMD heat-wave threshold, plains)
  3. WARM   - shift the whole temperature field by +0.5, +1, +2 ... C and
              recount -> the warming response falls out of the data, not a guess
  4. SAVE   - outputs/heatwave.json  (scenario.py reads this)

Heat-wave threshold:
  IMD declares a heat wave for plains at Tmax >= 40 C. Central India is a
  plains heat-wave region, so 40 C is the right absolute threshold.

Run (after `python scripts/preprocess.py --vars tmax`):
  python scripts/heatwave.py
"""

import os
import json
import numpy as np
import xarray as xr

# --- paths (match preprocess.py's naming) ---
PROC_DIR = os.path.join("data", "processed")
OUT = "outputs"

# --- heat-wave definition ---
THRESHOLD_C = 40.0                       # IMD plains heat-wave threshold
WARMING_STEPS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]  # C to test


def find_tmax_file():
    """Find the processed tmax NetCDF (central_tmax_<start>_<end>.nc)."""
    if not os.path.isdir(PROC_DIR):
        raise FileNotFoundError(
            f"No {PROC_DIR}/ folder. Run: python scripts/preprocess.py --vars tmax"
        )
    cands = [f for f in os.listdir(PROC_DIR)
             if f.startswith("central_tmax") and f.endswith(".nc")]
    if not cands:
        raise FileNotFoundError(
            f"No central_tmax_*.nc in {PROC_DIR}/. "
            "Run: python scripts/preprocess.py --vars tmax"
        )
    return os.path.join(PROC_DIR, sorted(cands)[-1])


def heatwave_days(tmax, valid, cell_is_land, delta=0.0):
    """
    Average heat-wave days per year at a typical land cell, after warming
    the whole field by `delta` degrees C.

    A heat-wave day at a cell = a day with (Tmax + delta) >= THRESHOLD_C.
    We count those per cell per year, then average over land cells and years.
    """
    hot = ((tmax + delta) >= THRESHOLD_C) & valid          # bool (time,lat,lon)
    per_year = hot.groupby("time.year").sum("time")        # (year,lat,lon)
    per_year = per_year.where(cell_is_land)                # drop ocean/edge cells
    return float(per_year.mean(["lat", "lon"]).mean("year"))


def main():
    path = find_tmax_file()
    print(f"Reading {path}")
    ds = xr.open_dataset(path)
    tmax = ds["tmax"]

    valid = tmax.notnull()                       # real readings (not masked no-data)
    cell_is_land = valid.mean("time") > 0.5      # cells with data most of the time

    n_years = len(np.unique(tmax["time"].dt.year))
    n_land = int(cell_is_land.sum())
    print(f"  years: {n_years} | land cells in box: {n_land} | threshold: {THRESHOLD_C} C\n")

    # --- baseline + warming response, straight from the data ---
    response = {}
    for d in WARMING_STEPS:
        response[d] = round(heatwave_days(tmax, valid, cell_is_land, d), 1)

    baseline = response[0.0]
    per_degC_local = round(response[1.0] - response[0.0], 1)  # slope over first degree

    print("=== Heat-wave days / year (Tmax >= 40 C), from IMD data ===")
    print(f"  baseline (today)        : {baseline} days/yr")
    for d in WARMING_STEPS[1:]:
        extra = round(response[d] - baseline, 1)
        print(f"  with +{d:.1f} C warming    : {response[d]} days/yr  ({extra:+.1f})")
    print(f"\n  local sensitivity       : ~{per_degC_local} extra days per +1 C "
          "(near today; rises faster with more warming)")

    # also report the region-wide view as a cross-check
    region_mean = tmax.mean(["lat", "lon"])
    region_hw = float((region_mean >= THRESHOLD_C).groupby("time.year").sum("time").mean())
    print(f"  cross-check (region-avg Tmax >= 40): {region_hw:.1f} days/yr "
          "(lower, as spatial averaging smooths local peaks)")

    # --- save for scenario.py ---
    os.makedirs(OUT, exist_ok=True)
    payload = {
        "region": "Central India",
        "threshold_c": THRESHOLD_C,
        "definition": "IMD plains heat wave: daily Tmax >= 40 C",
        "source": os.path.basename(path),
        "baseline_heatwave_days": baseline,
        "warming_response": {str(d): response[d] for d in WARMING_STEPS},
        "days_per_degC_near_baseline": per_degC_local,
    }
    with open(os.path.join(OUT, "heatwave.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {OUT}/heatwave.json")
    print("Now re-run scenario.py — it will use these real numbers.")


if __name__ == "__main__":
    main()