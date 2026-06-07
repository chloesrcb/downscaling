import re
import numpy as np
import pandas as pd
from pathlib import Path
from pyproj import Transformer
from scipy.spatial import cKDTree

from downscaling.config import TIME_COLS, SPATIAL_COLS


def find_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"No column found among {candidates}. Available columns:\n{list(cols)}")


def add_time_covariates(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Add cyclic time covariates."""
    df = df.copy()
    t = pd.to_datetime(df[time_col], utc=True)

    hour = t.dt.hour + t.dt.minute / 60
    doy = t.dt.dayofyear
    month = t.dt.month

    df["tod_sin"] = np.sin(2 * np.pi * hour / 24)
    df["tod_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df


def make_fine_grid_from_gauges(
    gauges: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    res_m: int,
    buffer_m: int,
) -> pd.DataFrame:
    """Build a regular Lambert-93 grid covering the gauge domain."""
    to_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    to_ll = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    x, y = to_l93.transform(
        gauges[lon_col].to_numpy(float),
        gauges[lat_col].to_numpy(float),
    )

    xmin = np.floor((x.min() - buffer_m) / res_m) * res_m + res_m / 2
    xmax = np.ceil((x.max() + buffer_m) / res_m) * res_m - res_m / 2
    ymin = np.floor((y.min() - buffer_m) / res_m) * res_m + res_m / 2
    ymax = np.ceil((y.max() + buffer_m) / res_m) * res_m - res_m / 2

    xs = np.arange(xmin, xmax + res_m, res_m)
    ys = np.arange(ymin, ymax + res_m, res_m)

    xx, yy = np.meshgrid(xs, ys)
    lon, lat = to_ll.transform(xx.ravel(), yy.ravel())

    return pd.DataFrame({
        "grid_id": np.arange(len(lon)),
        "station": [f"grid_{i}" for i in range(len(lon))],
        "lon_Y": lon,
        "lat_Y": lat,
        "x_l93": xx.ravel(),
        "y_l93": yy.ravel(),
    })


def read_comephore_wide(file_comephore: Path) -> pd.DataFrame:
    """Read COMEPHORE wide-format table."""
    com = pd.read_csv(file_comephore)

    time_col = find_col(com.columns, ["datetime", "time", "date", "dates"])
    com = com.rename(columns={time_col: "time"})
    com["time"] = pd.to_datetime(com["time"], utc=True)
    com.columns = [str(c) for c in com.columns]

    return com.sort_values("time").reset_index(drop=True)


def prepare_comephore_pixels(file_pixels: Path, comephore_cols: list[str]) -> pd.DataFrame:
    """Read COMEPHORE pixel coordinates and match identifiers to the wide table."""
    px = pd.read_csv(file_pixels)

    pixel_col = find_col(px.columns, ["pixel_id", "id_pixel", "id", "pixel", "cell_id", "pixel_name"])
    lon_col = find_col(px.columns, ["lon", "longitude", "Longitude", "lon_X"])
    lat_col = find_col(px.columns, ["lat", "latitude", "Latitude", "lat_X"])

    px = px.rename(columns={pixel_col: "pixel_id", lon_col: "lon_X", lat_col: "lat_X"})
    px["pixel_id"] = px["pixel_id"].astype(str)

    com_cols = set(map(str, comephore_cols))

    def match_pixel_id(v):
        v = str(v)
        if v in com_cols:
            return v
        if f"p{v}" in com_cols:
            return f"p{v}"
        if v.startswith("p") and v[1:] in com_cols:
            return v[1:]
        return v

    px["pixel_id"] = px["pixel_id"].map(match_pixel_id)

    to_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    px["x_l93"], px["y_l93"] = to_l93.transform(
        px["lon_X"].to_numpy(float),
        px["lat_X"].to_numpy(float),
    )

    return px[["pixel_id", "lon_X", "lat_X", "x_l93", "y_l93"]]


def attach_nearest_comephore_pixels(
    fine_grid: pd.DataFrame,
    pixels: pd.DataFrame,
    n_pixels: int = 9,
) -> pd.DataFrame:
    """Attach nearest COMEPHORE pixels to each fine-grid point."""
    tree = cKDTree(pixels[["x_l93", "y_l93"]].to_numpy(float))
    dist, idx = tree.query(fine_grid[["x_l93", "y_l93"]].to_numpy(float), k=n_pixels)

    out = fine_grid.copy()

    pix_ids = pixels["pixel_id"].to_numpy()
    pix_lon = pixels["lon_X"].to_numpy(float)
    pix_lat = pixels["lat_X"].to_numpy(float)

    for j in range(n_pixels):
        out[f"p{j + 1:02d}_id"] = pix_ids[idx[:, j]]
        out[f"p{j + 1:02d}_dist"] = dist[:, j]

    out["lon_X"] = pix_lon[idx[:, 0]]
    out["lat_X"] = pix_lat[idx[:, 0]]
    out["radar_gauge_distance"] = dist[:, 0]
    out["dist_XY"] = dist[:, 0]
    out["distance_XY"] = dist[:, 0]

    return out


def build_fine_grid_radar_predictors_5min(
    fine_grid: pd.DataFrame,
    comephore: pd.DataFrame,
    start_date: str,
    end_date: str,
    pred_freq: str = "5min",
) -> pd.DataFrame:
    """
    Build the grid-time prediction table at 5-min resolution.

    COMEPHORE covariates are hourly. For each 5-min prediction time, the
    corresponding hourly radar value is obtained by flooring the 5-min time
    to the current hour, then applying the -1h, 0h and +1h lags.
    """
    com = comephore.copy()
    com["time"] = pd.to_datetime(com["time"], utc=True)

    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")

    com = com[
        (com["time"] >= start - pd.Timedelta(hours=1)) &
        (com["time"] <= end + pd.Timedelta(hours=1))
    ].copy()

    target_times = pd.date_range(
        start=start,
        end=end,
        freq=pred_freq,
        inclusive="left",
        tz="UTC",
    )

    base = (
        pd.MultiIndex.from_product(
            [fine_grid["grid_id"], target_times],
            names=["grid_id", "time"],
        )
        .to_frame(index=False)
        .merge(fine_grid, on="grid_id", how="left")
    )

    pixel_cols = [f"p{j:02d}_id" for j in range(1, 10)]
    needed_pixels = sorted(set(base[pixel_cols].to_numpy().ravel().astype(str)))

    missing_pixels = [p for p in needed_pixels if p not in com.columns]
    if missing_pixels:
        raise ValueError(f"Missing COMEPHORE pixels in wide table: {missing_pixels[:20]}")

    com = com.set_index("time").sort_index()
    stacked_lookup = com[needed_pixels].stack()

    radar_hour = pd.to_datetime(base["time"], utc=True).dt.floor("h")

    for lag_label, lag_hours in [("dt-1h", -1), ("dt0h", 0), ("dt+1h", 1)]:
        lookup_time = radar_hour + pd.Timedelta(hours=lag_hours)

        for j in range(1, 10):
            idx_lookup = pd.MultiIndex.from_arrays([
                lookup_time,
                base[f"p{j:02d}_id"].astype(str),
            ])
            base[f"X_p{j:02d}_{lag_label}"] = stacked_lookup.reindex(idx_lookup).to_numpy(float)

    x_cols_cube = [c for c in base.columns if c.startswith("X_p")]
    base[x_cols_cube] = base[x_cols_cube].fillna(0.0)

    base["radar_mean"] = base[x_cols_cube].mean(axis=1)
    base["radar_max"] = base[x_cols_cube].max(axis=1)
    base["radar_sum"] = base[x_cols_cube].sum(axis=1)

    dt0_cols = [c for c in x_cols_cube if "dt0h" in c]
    base["radar_mean_dt0h"] = base[dt0_cols].mean(axis=1)
    base["radar_max_dt0h"] = base[dt0_cols].max(axis=1)
    base["radar_sum_dt0h"] = base[dt0_cols].sum(axis=1)
    base["radar_central_dt0h"] = base["X_p01_dt0h"]

    base = add_time_covariates(base, "time")

    base["Y_obs"] = 1.0
    base["is_fake_grid"] = True

    print("Fine grid points:", fine_grid["grid_id"].nunique())
    print("5-min prediction times:", len(target_times))
    print("Prediction rows:", len(base))

    return base


def load_raw_downscaling_table(path: str | None = None, sep: str = ";") -> pd.DataFrame:
    """
    Load the raw OMSEV/COMEPHORE downscaling table.
    """
    if path is None:
        from downscaling.paths import DOWNSCALING_TABLE
        path = DOWNSCALING_TABLE

    df = pd.read_csv(path, sep=sep)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    return df


def get_x_cols27_downscaling(df: pd.DataFrame) -> list[str]:
    """
    Get the 27 radar cube columns named X_pXX_dt...
    Ordered by pixel number and time lag.
    """
    pat = re.compile(r"^X_p(\d{2})_dt(-1h|0h|\+1h)$")
    cols = [c for c in df.columns if pat.match(c)]

    order_dt = {"-1h": 0, "0h": 1, "+1h": 2}

    def key(c):
        m = pat.match(c)
        return int(m.group(1)), order_dt[m.group(2)]

    return sorted(cols, key=key)


def prepare_modeling_dataframe(
    df_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """
    Prepare the dataframe used for model fitting.

    This function:
    - converts time;
    - keeps positive OMSEV observations;
    - creates time covariates;
    - creates radar summary covariates;
    - filters observations where OMSEV and radar are both positive.
    """
    df = df_raw.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)

    x_cols27 = get_x_cols27_downscaling(df)

    if len(x_cols27) == 0:
        raise ValueError("No radar columns matching X_pXX_dt... were found.")

    df = df.loc[df["Y_obs"].notna() & (df["Y_obs"] > 0)].copy()
    df[x_cols27] = df[x_cols27].apply(pd.to_numeric, errors="coerce")

    df["hour"] = df["time"].dt.hour.astype(int)
    df["minute"] = df["time"].dt.minute.astype(int)
    df["month"] = df["time"].dt.month.astype(int)
    df["year"] = df["time"].dt.year.astype(int)

    tod = df["hour"] * 60 + df["minute"]
    doy = df["time"].dt.dayofyear.astype(float)

    df["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    df["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    df["doy_sin"] = np.sin(2 * np.pi * (doy - 1) / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * (doy - 1) / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    X_block = df[x_cols27].to_numpy(dtype=float)

    df["radar_max"] = np.nanmax(X_block, axis=1)
    df["radar_mean"] = np.nanmean(X_block, axis=1)
    df["radar_sum"] = np.nansum(X_block, axis=1)

    x_cols_dt0h = [c for c in x_cols27 if c.endswith("dt0h")]

    if len(x_cols_dt0h) == 0:
        raise ValueError("No current-hour radar columns ending with dt0h were found.")

    X_block_dt0h = df[x_cols_dt0h].to_numpy(dtype=float)

    df["radar_max_dt0h"] = np.nanmax(X_block_dt0h, axis=1)
    df["radar_mean_dt0h"] = np.nanmean(X_block_dt0h, axis=1)
    df["radar_sum_dt0h"] = np.nansum(X_block_dt0h, axis=1)

    central_col = "X_p01_dt0h"

    if central_col in df.columns:
        df["radar_central_dt0h"] = df[central_col].astype(float)
    else:
        df["radar_central_dt0h"] = df["radar_max_dt0h"]

    df["corres"] = (
        (df["Y_obs"] > 0)
        & (np.nansum(X_block, axis=1) > 0)
    ).astype(int)

    keep_cols = [
        "time",
        "station",
        "Y_obs",
        "year",
        "month",
        "hour",
        "minute",
        *TIME_COLS,
        *SPATIAL_COLS,
        "radar_max",
        "radar_mean",
        "radar_sum",
        "radar_max_dt0h",
        "radar_mean_dt0h",
        "radar_sum_dt0h",
        "radar_central_dt0h",
        *x_cols27,
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df.loc[df["corres"] == 1, keep_cols].copy()

    x_cols_all = list(dict.fromkeys(
        TIME_COLS
        + SPATIAL_COLS
        + [
            "radar_max",
            "radar_mean",
            "radar_sum",
            "radar_max_dt0h",
            "radar_mean_dt0h",
            "radar_sum_dt0h",
            "radar_central_dt0h",
        ]
        + x_cols27
    ))

    x_cols_all = [c for c in x_cols_all if c in df.columns]

    return df, x_cols27, x_cols_dt0h, x_cols_all


def make_covariate_sets(
    x_cols27: list[str],
    x_cols_dt0h: list[str],
    x_cols_all: list[str],
) -> dict[str, list[str]]:
    """
    Build named covariate sets for NN models.
    """
    radar_summary_all = [
        "radar_max",
        "radar_mean",
        "radar_sum",
        "radar_max_dt0h",
        "radar_mean_dt0h",
        "radar_sum_dt0h",
        "radar_central_dt0h",
    ]

    radar_summary_all = [c for c in radar_summary_all if c in x_cols_all]

    local_pixels = [
        c for c in [
            "X_p01_dt0h",
            "X_p02_dt0h",
            "X_p03_dt0h",
            "X_p04_dt0h",
            "X_p05_dt0h",
        ]
        if c in x_cols_all
    ]

    x_sets = {
        "central_only": ["radar_central_dt0h"],
        "summary_dt0h": [
            "radar_max_dt0h",
            "radar_mean_dt0h",
            "radar_sum_dt0h",
            "radar_central_dt0h",
        ],
        "local_pixels_dt0h": local_pixels,
        "all_pixels_dt0h": x_cols_dt0h,
        "radar_summaries": radar_summary_all,
        "radar_all": radar_summary_all + x_cols27,
        "radar_time": TIME_COLS + radar_summary_all + x_cols27,
        "radar_time_space": x_cols_all,
    }

    x_sets = {
        name: list(dict.fromkeys([c for c in cols if c in x_cols_all]))
        for name, cols in x_sets.items()
    }

    x_sets = {
        name: cols
        for name, cols in x_sets.items()
        if len(cols) > 0
    }

    return x_sets
