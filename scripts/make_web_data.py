"""
make_web_data.py  --  feed the dashboard with real numbers
----------------------------------------------------------
Reads outputs/scenario.json (+ outputs/forecast.json if present) and writes
web/vayu-data.js, right next to vayu-dashboard.html so the double-clicked page
picks it up.

Why a .js file and not the .json directly?
  Opening the dashboard by double-clicking (file://) blocks JavaScript from
  fetch()-ing local .json. A <script> tag is allowed to read a local file,
  so we wrap the data in one.

Run from the repo root after scenario.py / forecast.py:
  python3 scripts/make_web_data.py
"""

import os
import json
from datetime import datetime

OUT_DIR = "outputs"     # where the pipeline writes its JSON
WEB_DIR = "web"         # where vayu-dashboard.html lives


def _load(name):
    with open(os.path.join(OUT_DIR, name)) as f:
        return json.load(f)


def _label(date_str):
    """'2024-01-01' -> 'Jan 1' (no leading zero)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return f"{dt:%b} {dt.day}"


def main():
    # --- scenario (required) ---
    sc = _load("scenario.json")
    b = sc["baseline"]
    curves = sc.get("curves", {})
    data = {
        "scenario": {
            "annualRain": b["annual_rain_mm"],
            "baseHeatwave": b["heatwave_days"],
            "baseWarmNights": b.get("warm_nights"),
            "heatwaveCurve": curves.get("heatwave_days"),
            "warmNightsCurve": curves.get("warm_nights"),
        }
    }

    # --- forecast (optional) ---
    fc_path = os.path.join(OUT_DIR, "forecast.json")
    if os.path.exists(fc_path):
        arr = _load("forecast.json")["forecast"]
        data["forecast"] = {
            "days": [_label(d["date"]) for d in arr],
            "rain": [d["rain"] for d in arr],
            "high": [d["high"] for d in arr],
            "low":  [d.get("low", 0) for d in arr],
        }

    # --- model skill metrics (optional) ---
    m_path = os.path.join(OUT_DIR, "metrics.json")
    if os.path.exists(m_path):
        m = _load("metrics.json")
        data["model"] = {
            "gain": m.get("gain_pct"),
            "rmseGuess": m.get("rmse_persistence"),
            "rmseAI": m.get("rmse_ai"),
            "trainDays": m.get("train_days"),
            "yearsTested": m.get("years_tested"),
        }

    os.makedirs(WEB_DIR, exist_ok=True)
    path = os.path.join(WEB_DIR, "vayu-data.js")
    with open(path, "w") as f:
        f.write("window.VAYU_DATA = " + json.dumps(data, indent=2) + ";\n")

    print(f"Wrote {path}")
    print(f"  baseHeatwave   = {data['scenario']['baseHeatwave']}")
    print(f"  baseWarmNights = {data['scenario']['baseWarmNights']}")
    if "forecast" in data:
        print(f"  forecast days  = {len(data['forecast']['rain'])} "
              f"({data['forecast']['days'][0]} -> {data['forecast']['days'][-1]})")
    else:
        print("  forecast       = (no forecast.json found, skipped)")
    print("Double-click web/vayu-dashboard.html to see it.")


if __name__ == "__main__":
    main()