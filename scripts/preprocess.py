"""
preprocess.py
-------------
Turn raw IMD data into analysis-ready files for the Central India pilot region.

It does four things, per variable:
  1. LOAD   - open cached files, or download the year range if missing
  2. CLEAN  - mask IMD's no-data sentinels -> NaN
  3. CROP   - cut down to the Central India box
  4. SAVE   - write a tidy NetCDF + a quick-look figure to confirm

Central India pilot box:  lat 18-26 N , lon 74-84 E   (the monsoon core)

Usage:
  python scripts/preprocess.py                                   # rain, 2005-2023
  python scripts/preprocess.py --vars rain tmax tmin --start 2000 --end 2023
"""

import argparse
import os
import imdlib as imd
import matplotlib.pyplot as plt

RAW_DIR = os.path.join("data", "raw")
PROC_DIR = os.path.join("data", "processed")
FIG_DIR = "outputs"


LAT_MIN, LAT_MAX = 18, 26
LON_MIN, LON_MAX = 74, 84

# IMD fills "no data here" cells with these placeholder values
NODATA = {"rain": -999.0, "tmax": 99.9, "tmin": 99.9}
LABELS = {"rain": "Rainfall (mm/day)", "tmax": "Max temp (°C)", "tmin": "Min temp (°C)"}
CMAPS = {"rain": "Blues", "tmax": "RdYlBu_r", "tmin": "RdYlBu_r"}


def load_variable(var, start, end):
    
    os.makedirs(RAW_DIR, exist_ok=True)
    try:
        data = imd.open_data(var, start, end, "yearwise", file_dir=RAW_DIR)
        print(f"  [{var}] using cached files in {RAW_DIR}")
    except Exception:
        print(f"  [{var}] downloading {start}-{end} from IMD (one-time, be patient)...")
        data = imd.get_data(var, start, end, fn_format="yearwise", file_dir=RAW_DIR)
    return data.get_xarray()


def process(var, start, end):
    print(f"\n=== {var} ({start}-{end}) ===")
    ds = load_variable(var, start, end)

    ds = ds.where(ds[var] != NODATA[var])


    ds = ds.sortby("lat").sortby("lon")
    ds = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))

    # quick sanity report
    nt = ds.sizes["time"]
    nlat = ds.sizes["lat"]
    nlon = ds.sizes["lon"]
    pct_missing = float(ds[var].isnull().mean()) * 100
    print(f"  cropped shape : time={nt}, lat={nlat}, lon={nlon}")
    print(f"  missing cells : {pct_missing:.1f}%  (ocean/edges over the box)")

   
    os.makedirs(PROC_DIR, exist_ok=True)
    out = os.path.join(PROC_DIR, f"central_{var}_{start}_{end}.nc")
    ds.to_netcdf(out)
    print(f"  saved         : {out}")

    
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure(figsize=(5.5, 5))
    ds[var].mean("time").plot(cmap=CMAPS[var], cbar_kwargs={"label": LABELS[var]})
    plt.title(f"Central India — mean {var} ({start}-{end})")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.tight_layout()
    fig = os.path.join(FIG_DIR, f"central_{var}_mean.png")
    plt.savefig(fig, dpi=130)
    plt.close()
    print(f"  figure        : {fig}")


def main():
    p = argparse.ArgumentParser(description="Preprocess IMD data for Central India")
    p.add_argument("--vars", nargs="+", default=["rain"],
                   choices=["rain", "tmax", "tmin"])
    p.add_argument("--start", type=int, default=2005, help="start year")
    p.add_argument("--end", type=int, default=2023, help="end year")
    args = p.parse_args()

    print(f"Pilot region: lat {LAT_MIN}-{LAT_MAX} N, lon {LON_MIN}-{LON_MAX} E (Central India)")
    for v in args.vars:
        try:
            process(v, args.start, args.end)
        except Exception as e:
            print(f"  !! failed for {v}: {e}")

    print("\nDone. Cropped, cleaned files are in data/processed/ — ready for baselines.")


if __name__ == "__main__":
    main()