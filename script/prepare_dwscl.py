
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
if __name__ == "__main__":
    data_folder = os.environ.get("DATA_FOLDER", "../phd_extremes/data/")
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    paths = preprocess.Paths(
        filename_com=os.path.join(data_folder, "comephore/rebuild_clean/comephore_2008_2025_within5km.csv"),
        filename_loc_px=os.path.join(data_folder, "comephore/rebuild_clean/coords_pixels_within5km.csv"),
        filename_rain_rdata=os.path.join(data_folder, "omsev/omsev_5min/rain_mtp_5min_2019_2024.csv"),
        filename_loc_gauges=os.path.join(data_folder, "omsev/loc_rain_gauges.csv"),
        output_file=os.path.join(data_folder, "downscaling/downscaling_table_named_2019_2024.csv"),
    )

    preprocess.build_table(
        paths=paths,
        start="2019-01-01",
        end="2025-01-01",
        radius_m=1500.0,
        n_feat=27,
        n_jobs=max(1, n_jobs),
        chunksize_com=20000,
    )

# %%
# comephore = pd.read_csv(paths.filename_com, sep=",", header=0)

# # plot comephore pixel p100
# import matplotlib.pyplot as plt
# comephore["datetime"] = to_utc(comephore["datetime"])
# plt.plot(comephore["datetime"], comephore["p81"])
# plt.title("COMEPHORE pixel p100 over time")
# plt.xlabel("Time")
# plt.ylabel("p100 value")
# plt.grid()
# plt.show()