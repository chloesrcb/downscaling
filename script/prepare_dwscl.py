
#%%
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pyreadr
from pyproj import Transformer
from joblib import Parallel, delayed

import downscaling.preprocess as preprocess

#%%
data_folder = os.environ.get("DATA_FOLDER", "../../phd_extremes/data/")
n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

paths = preprocess.Paths(
        filename_com=os.path.join(data_folder, "comephore/rebuild_clean/comephore_2008_2025_within5km.csv"),
        filename_loc_px=os.path.join(data_folder, "comephore/rebuild_clean/coords_pixels_within5km.csv"),
        filename_rain_rdata=os.path.join(data_folder, "omsev/omsev_5min/rain_mtp_5min_2019_2025.csv"),
        filename_loc_gauges=os.path.join(data_folder, "omsev/loc_rain_gauges.csv"),
        output_file=os.path.join(data_folder, "downscaling/downscaling_table_named_2019_2025.csv"),
)

# %%
rain = preprocess.load_omsev(paths, start="2019-09-01", end="2026-01-01")
loc_gauges = pd.read_csv(paths.filename_loc_gauges)
loc_px = pd.read_csv(paths.filename_loc_px)

#%%
print("\n=== OMSEV ===")
print("rows:", len(rain))
print("stations:", rain.columns)

#%%
print("\n=== LOC GAUGES ===")
print("rows:", len(loc_gauges))
print(loc_gauges.columns)
#%%
print("stations loc:", loc_gauges["Station"].nunique())
print(loc_gauges["Station"].unique())

#%%

# remove "brives", "hydro", "cines" stations from loc_gauges
# loc_gauges = loc_gauges[~loc_gauges["Station"].isin(["brives", "hydro", "cines"])]
# # remove it from rain as well
# rain = rain.drop(columns=["brives", "hydro", "cines"])

#%%
# plot loc_gauges and loc_px on a map

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(loc_px["Longitude"], loc_px["Latitude"], c="blue", label="Radar pixels", alpha=0.5, s=10)
plt.scatter(loc_gauges["Longitude"], loc_gauges["Latitude"], c="red", label="Rain gauges", s=50
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Radar pixels and rain gauges locations")
plt.legend()
plt.grid()

#%%

preprocess.build_table(
        paths=paths,
        start="2019-09-01",
        end="2026-01-01",
        radius_m=1500.0,
        n_feat=27,
        n_jobs=max(1, n_jobs),
        chunksize_com=20000,
)

# %%
df_final = pd.read_csv(paths.output_file, sep=";")
print("rows:", len(df_final))
print("columns:", df_final.columns.tolist())

#%%
station_col = "station"

print("stations:", df_final[station_col].nunique())
print(sorted(df_final[station_col].unique()))
print(df_final[station_col].value_counts().sort_index())

#%%
df_final.groupby("station")[
    ["lon_Y", "lat_Y", "lon_X", "lat_X", "dist_gauge_to_pixel_m"]
].first().sort_index()


#%%
# read downscaling_table_named_2019_2025.csv
df_check = pd.read_csv(os.path.join(data_folder, "downscaling/downscaling_table_named_2019_2025.csv"), sep=";")

#%%
df_coords = df_final.groupby("station")[["lon_Y", "lat_Y", "lon_X", "lat_X"]].first()

#%%
# plot station names on lon_Y, lat_Y 
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(df_coords["lon_Y"], df_coords["lat_Y"], c="red", label="Rain gauges", s=50)
for station, row in df_coords.iterrows():
    plt.text(row["lon_Y"], row["lat_Y"], station, fontsize=8, ha="right", va="bottom")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Rain gauges locations with station names")
plt.legend()
plt.grid()

#%% for cines sites plot the radar lon_X, lat_X on the map with a different color
point_pixel = df_coords.loc[df_coords.index.str.contains("cines"), ["lon_X", "lat_X"]]
plt.figure(figsize=(8, 6))
plt.scatter(df_coords["lon_Y"], df_coords["lat_Y"], c="red", label="Rain gauges", s=50)
plt.scatter(point_pixel["lon_X"], point_pixel["lat_X"], c="green", label="Radar pixels for cines sites", s=50)
for station, row in df_coords.iterrows():
    plt.text(row["lon_Y"], row["lat_Y"], station, fontsize=8, ha="right", va="bottom")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Rain gauges locations with station names and radar pixels for cines sites")
plt.legend()
plt.grid()

#%%
# get name of the radar pixel for cines sites
df_coords.loc[df_coords.index.str.contains("cines"), ["lon_X", "lat_X"]]
# corresponding radar pixel in loc_px
loc_px.loc[(loc_px["Longitude"] == df_coords.loc[df_coords.index.str.contains("cines"), "lon_X"].values[0]) & (loc_px["Latitude"] == df_coords.loc[df_coords.index.str.contains("cines"), "lat_X"].values[0])]

#%%
# plot on map all the radar pixel with the name of the pixel and the all rain gauges
xmin, xmax = 3.83, 3.88
ymin, ymax = 43.61, 43.65

plt.figure(figsize=(8, 6))

plt.scatter(
    loc_px["Longitude"], loc_px["Latitude"],
    c="blue", label="Radar pixels", alpha=0.5, s=10
)

plt.scatter(
    loc_gauges["Longitude"], loc_gauges["Latitude"],
    c="red", label="Rain gauges", s=50
)

for station, row in df_coords.iterrows():
    if xmin <= row["lon_Y"] <= xmax and ymin <= row["lat_Y"] <= ymax:
        plt.text(
            row["lon_Y"], row["lat_Y"],
            station,
            fontsize=8,
            ha="right",
            va="bottom"
        )

for _, row in loc_px.iterrows():
    if xmin <= row["Longitude"] <= xmax and ymin <= row["Latitude"] <= ymax:
        plt.text(
            row["Longitude"], row["Latitude"],
            row["pixel_name"],
            fontsize=6,
            ha="left",
            va="bottom"
        )

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.title("Radar pixels and rain gauges locations with names")
plt.legend()
plt.grid()
plt.show()

