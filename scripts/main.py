import os
import imdlib as imd
import matplotlib.pyplot as plt

os.makedirs("data/raw", exist_ok=True)

data = imd.get_data("rain", 2022, 2023, fn_format="yearwise", file_dir="data/raw")
ds = data.get_xarray()
ds = ds.where(ds["rain"] != -999.0)   # mask no-data (ocean/edges) -> NaN

print("Loaded:", ds.sizes)



plt.figure()   # <- start a fresh figure
ds["rain"].sel(time=slice("2023-06-01", "2023-08-31")).mean("time").plot(cmap="Blues")
plt.title("Mean monsoon rainfall — JJA 2023")
plt.show()