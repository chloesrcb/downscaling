
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
        filename_rain_rdata=os.path.join(data_folder, "omsev/omsev_5min/rain_mtp_5min_2019_2024.csv"),
        filename_loc_gauges=os.path.join(data_folder, "omsev/loc_rain_gauges.csv"),
        output_file=os.path.join(data_folder, "downscaling/downscaling_table_named_2019_2024.csv"),
)

# %%
rain = preprocess.load_omsev(paths, start="2019-01-01", end="2025-01-01")
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
print("stations loc:", loc_gauges["Station"].nunique())
print(sorted(loc_gauges["Station"].unique()))

# change name of Station
loc_gauges["Station"] = ["iem", "mse", "poly", "um", "cefe", "cnrs", "crbm",
                      "archie", "archiw", "um35",
                      "chu1", "chu2", "chu3", "chu4", "chu5", "chu6", "chu7",
                      "cines", "brives", "hydro"]

# remove "brives", "hydro", "cines" stations from loc_gauges
loc_gauges = loc_gauges[~loc_gauges["Station"].isin(["brives", "hydro", "cines"])]
# remove it from rain as well
rain = rain.drop(columns=["brives", "hydro", "cines"])

#%%

preprocess.build_table(
        paths=paths,
        start="2019-09-01",
        end="2025-01-01",
        radius_m=1500.0,
        n_feat=27,
        n_jobs=max(1, n_jobs),
        chunksize_com=20000,
)

# %%
df_final = pd.read_csv(paths.output_file, sep=";")

print("\n=== TABLE FINALE ===")
print("rows:", len(df_final))
print("columns:", df_final.columns.tolist())

#%%
station_col = "station"

print("stations finales:", df_final[station_col].nunique())
print(sorted(df_final[station_col].unique()))
print(df_final[station_col].value_counts().sort_index())

# %%

# rain = preprocess.load_omsev(paths, start="2020-01-01", end="2020-01-10")

# location_gauges, loc_px, station_pixels = build_station_radius_pixels(
#     paths=paths,
#     rain_hsm=rain,
#     radius_m=1500.0,
# )
# #%%
# print("\n=== Pixel coordinate sanity check ===")
# print(loc_px.head())
# print(loc_px[["lon_X", "lat_X", "X_m", "Y_m"]].describe())

# #%%
# print("\n=== Gauge / central pixel check ===")
# cols = ["station", "Longitude", "Latitude", "closest_pixel", "lon_X", "lat_X", "X_m", "Y_m"]
# print(location_gauges[cols].sort_values("station"))

# print("\nUnique central pixels:")
# print(location_gauges["closest_pixel"].value_counts())

# #%%
# rows = []

# for _, row in location_gauges.iterrows():
#     st = row["station"]
#     pix_list = station_pixels[st]

#     rows.append({
#         "station": st,
#         "closest_pixel": row["closest_pixel"],
#         "p01_pixel": pix_list[0],
#         "p01_is_closest": pix_list[0] == row["closest_pixel"],
#         "n_pixels": len(pix_list),
#     })

# check_p01 = pd.DataFrame(rows)
# print(check_p01.sort_values("station"))