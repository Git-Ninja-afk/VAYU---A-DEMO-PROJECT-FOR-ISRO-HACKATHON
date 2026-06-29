"""
scenario.py  --  the "what-if" engine
-------------------------------------
Translate climate scenarios into REAL-WORLD IMPACTS for Central India.

Inputs (the dashboard sliders):
    rain_change_pct : change in seasonal/annual rainfall  (e.g. -30 = 30% less)
    warming_c       : warming applied                     (e.g. +2.0 degC)

Outputs (impact tiles):
    projected rainfall (mm)  -- grounded in real IMD rainfall data
    drought risk category    -- via IMD's official rainfall-departure thresholds
    heatwave days / year     -- REAL: from IMD tmax (scripts/heatwave.py)
    warm nights / year       -- REAL: from IMD tmin (scripts/warm_nights.py)

Heat impacts are read from outputs/heatwave.json and outputs/warm_nights.json.
If those files are missing, the engine falls back to a clearly-labeled guess
so it still runs -- but run heatwave.py / warm_nights.py to make them real.

Also saves outputs/scenario.json (baseline + examples + curves for the dashboard).

Run:  python3 scripts/scenario.py
"""

import os, json
import numpy as np
import pandas as pd
import xarray as xr

PROC = os.path.join("data", "processed", "central_rain_2005_2023.nc")
OUT = "outputs"


# =====================================================================
#  Heat impacts: load REAL curves from the temperature scripts' output
# =====================================================================
def _load_curve(filename, baseline_key, slope_key):
    """
    Load a {warming_c -> days/year} response curve produced by heatwave.py
    or warm_nights.py. Returns (curve_dict, baseline, slope, is_real).
    If the file is missing/broken, returns (None, ...) so callers fall back.
    """
    path = os.path.join(OUT, filename)
    try:
        with open(path) as f:
            data = json.load(f)
        curve = {float(k): float(v) for k, v in data["warming_response"].items()}
        baseline = float(data[baseline_key])
        slope = float(data[slope_key])
        return curve, baseline, slope, True
    except (FileNotFoundError, KeyError, ValueError):
        return None, None, None, False


# --- heatwave (Tmax >= 40C) ---
_HW_CURVE, _HW_BASE, _HW_SLOPE, HEATWAVE_IS_REAL = _load_curve(
    "heatwave.json", "baseline_heatwave_days", "days_per_degC_near_baseline"
)
if not HEATWAVE_IS_REAL:                       # fallback guess (run heatwave.py!)
    _HW_BASE, _HW_SLOPE = 18.0, 6.0

# --- warm nights (Tmin >= 25C) ---
_WN_CURVE, _WN_BASE, _WN_SLOPE, WARM_NIGHTS_IS_REAL = _load_curve(
    "warm_nights.json", "baseline_warm_nights", "nights_per_degC_near_baseline"
)
if not WARM_NIGHTS_IS_REAL:                    # fallback guess (run warm_nights.py!)
    _WN_BASE, _WN_SLOPE = 12.0, 5.0

# exposed for the rest of the file / dashboard
BASE_HEATWAVE_DAYS = _HW_BASE
HEATWAVE_PER_DEGC = _HW_SLOPE
BASE_WARM_NIGHTS = _WN_BASE
WARM_NIGHTS_PER_DEGC = _WN_SLOPE


def _interp(curve, base, slope, warming_c):
    """Days/year at a given warming: use the real curve if we have it
    (captures the acceleration), else a straight-line fallback."""
    if curve is None:
        return base + slope * warming_c
    xs = sorted(curve)
    ys = [curve[x] for x in xs]
    return float(np.interp(warming_c, xs, ys))   # clamps at the curve ends


def heatwave_for(warming_c):
    return _interp(_HW_CURVE, _HW_BASE, _HW_SLOPE, warming_c)


def warm_nights_for(warming_c):
    return _interp(_WN_CURVE, _WN_BASE, _WN_SLOPE, warming_c)


# =====================================================================
#  Rainfall impact: IMD-style drought classification
# =====================================================================
def drought_risk(departure_pct):
    """Classify rainfall departure using IMD-style categories."""
    if departure_pct >= 20:
        return "Excess (flood watch)", "high", "surplus rainfall, runoff/flood risk"
    elif departure_pct > -20:
        return "Normal", "low", "water balance near normal"
    elif departure_pct > -60:
        return "Deficient (drought)", "high", "rainfall shortfall, crop & water stress"
    else:
        return "Large deficit (severe drought)", "severe", "acute deficit, water emergency"


def simulate(rain_change_pct, warming_c, base):
    """Return the impact tiles for one scenario."""
    new_rain = base["annual_rain_mm"] * (1 + rain_change_pct / 100)
    cat, level, note = drought_risk(rain_change_pct)

    base_hw = round(heatwave_for(0.0))
    heatwave = round(heatwave_for(warming_c))

    base_wn = round(warm_nights_for(0.0))
    warm_nights = round(warm_nights_for(warming_c))

    return {
        "rain_change_pct": rain_change_pct,
        "warming_c": warming_c,
        "projected_annual_rain_mm": round(new_rain),
        "rain_delta_mm": round(new_rain - base["annual_rain_mm"]),
        "drought_category": cat,
        "drought_level": level,
        "drought_note": note,
        "heatwave_days": heatwave,
        "heatwave_delta": heatwave - base_hw,
        "warm_nights": warm_nights,
        "warm_nights_delta": warm_nights - base_wn,
    }


def main():
    # ---------- compute REAL baseline normals from the data ----------
    ds = xr.open_dataset(PROC)
    daily = ds["rain"].mean(dim=["lat", "lon"]).to_series()
    daily.index = pd.to_datetime(daily.index)

    # annual rainfall normal = average of yearly totals
    annual_totals = daily.groupby(daily.index.year).sum()
    annual_rain = float(annual_totals.mean())

    # monsoon (Jun-Sep) normal
    monsoon = daily[daily.index.month.isin([6, 7, 8, 9])]
    monsoon_total = float(monsoon.groupby(monsoon.index.year).sum().mean())

    base = {"annual_rain_mm": annual_rain, "monsoon_rain_mm": monsoon_total}

    hw_tag = "IMD tmax data" if HEATWAVE_IS_REAL else "GUESS - run heatwave.py"
    wn_tag = "IMD tmin data" if WARM_NIGHTS_IS_REAL else "GUESS - run warm_nights.py"

    print("=== Central India climate baseline (from your data) ===")
    print(f"  Normal annual rainfall  : {annual_rain:.0f} mm/year")
    print(f"  Normal monsoon (JJAS)   : {monsoon_total:.0f} mm")
    print(f"  Baseline heatwave days  : {round(BASE_HEATWAVE_DAYS)} /yr  ({hw_tag})")
    print(f"  Baseline warm nights    : {round(BASE_WARM_NIGHTS)} /yr  ({wn_tag})\n")

    # ---------- run a few example scenarios ----------
    examples = [
        (0, 0.0),      # no change
        (-30, 0.0),    # 30% less rain
        (0, 2.0),      # +2 degC warming
        (-30, 2.0),    # drought + warming combo
        (20, 0.0),     # 20% more rain
    ]
    print("=== What-if scenarios ===")
    sims = []
    for rpct, w in examples:
        s = simulate(rpct, w, base)
        sims.append(s)
        print(f"\n  rain {rpct:+d}% , warming +{w:.1f}C")
        print(f"    projected rainfall : {s['projected_annual_rain_mm']} mm "
              f"({s['rain_delta_mm']:+d} mm)")
        print(f"    drought risk       : {s['drought_category']}  ({s['drought_level']})")
        print(f"    heatwave days/yr   : {s['heatwave_days']} ({s['heatwave_delta']:+d})")
        print(f"    warm nights/yr     : {s['warm_nights']} ({s['warm_nights_delta']:+d})")

    # ---------- save scenario.json (dashboard feed) ----------
    os.makedirs(OUT, exist_ok=True)
    payload = {
        "region": "Central India",
        "baseline": {
            "annual_rain_mm": round(annual_rain),
            "monsoon_rain_mm": round(monsoon_total),
            "heatwave_days": round(BASE_HEATWAVE_DAYS),
            "warm_nights": round(BASE_WARM_NIGHTS),
        },
        "sensitivity": {
            "heatwave_days_per_degC": round(HEATWAVE_PER_DEGC, 1),
            "warm_nights_per_degC": round(WARM_NIGHTS_PER_DEGC, 1),
        },
        "sources": {
            "rainfall": "IMD gridded rainfall (0.25deg)",
            "heatwave": "IMD tmax >=40C" if HEATWAVE_IS_REAL else "estimate",
            "warm_nights": "IMD tmin >=25C" if WARM_NIGHTS_IS_REAL else "estimate",
        },
        # full warming-response curves so the dashboard can plot them
        "curves": {
            "heatwave_days": {str(w): round(heatwave_for(w), 1)
                              for w in [0, 0.5, 1, 1.5, 2, 2.5, 3, 4]},
            "warm_nights": {str(w): round(warm_nights_for(w), 1)
                            for w in [0, 0.5, 1, 1.5, 2, 2.5, 3, 4]},
        },
        "examples": sims,
    }
    with open(os.path.join(OUT, "scenario.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {OUT}/scenario.json")
    if not HEATWAVE_IS_REAL or not WARM_NIGHTS_IS_REAL:
        print("NOTE: some heat numbers are still GUESSES - "
              "run heatwave.py / warm_nights.py to make them real.")


if __name__ == "__main__":
    main()