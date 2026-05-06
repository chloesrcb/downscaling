from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyreadr
from pyproj import Transformer
from joblib import Parallel, delayed

def to_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True)

def _search_window_indices(sorted_ns: np.ndarray, t_ns: int, half_window_ns: int) -> tuple[int, int]:
    """
    Returns [i0, i1) indices in sorted_ns such that:
    |sorted_ns - t_ns| <= half_window_ns
    i0 inclusive, i1 exclusive
    """
    left = t_ns - half_window_ns
    right = t_ns + half_window_ns
    i0 = int(np.searchsorted(sorted_ns, left, side="left"))
    i1 = int(np.searchsorted(sorted_ns, right, side="right"))
    return i0, i1

def make_X_names(n_feat: int = 27) -> list[str]:
    """Names on matrix [time, pix] with 3 fixed times (-1h,0,+1h)."""
    dt_labels = ["-1h", "0h", "+1h"]
    names = []
    k = 0
    p = 1
    while k < n_feat:
        for dt in dt_labels:
            if k >= n_feat:
                break
            names.append(f"X_p{p:02d}_dt{dt}")
            k += 1
        p += 1
    return names


@dataclass
class Paths:
    filename_com: str
    filename_loc_px: str
    filename_rain_rdata: str
    filename_loc_gauges: str
    output_file: str


# Load OMSEV and restrict to a date range
def load_omsev(paths: Paths, start: str, end: str) -> pd.DataFrame:
    r = pd.read_csv(paths.filename_rain_rdata, sep=",", header=0, index_col=0)
    rain_hsm = r.copy()
    # rain_hsm.columns = ["dates"] + list(rain_hsm.columns[1:])
    rain_hsm["dates"] = to_utc(rain_hsm.index)

    gauge_cols = [c for c in rain_hsm.columns if c != "dates"]
    rain_hsm = rain_hsm.loc[~rain_hsm[gauge_cols].isna().all(axis=1)].copy()

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    rain_hsm = rain_hsm[(rain_hsm["dates"] >= start_ts) & (rain_hsm["dates"] < end_ts)].copy()
    return rain_hsm


def nearest_time_index_one(target_ns: int, com_ns: np.ndarray) -> int:
    """Nearest index in sorted com_ns to target_ns."""
    pos = int(np.searchsorted(com_ns, target_ns, side="left"))
    if pos <= 0:
        return 0
    if pos >= len(com_ns):
        return len(com_ns) - 1
    pos0 = pos - 1
    return pos0 if abs(com_ns[pos0] - target_ns) <= abs(com_ns[pos] - target_ns) else pos

# Station -> closest pixel + pixels within radius
def build_station_radius_pixels(
    paths: Paths,
    rain_hsm: pd.DataFrame,
    radius_m: float = 1500.0,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """
    Logic:
      - find closest pixel to gauge
      - select all pixels within radius_m of that closest pixel (in meters)
    Returns:
      - location_gauges filtered to stations present in rain_hsm
      - loc_px with projected coords
      - station_pixels: station -> list of pixel_name (in loc_px file order)
    """
    loc_px = pd.read_csv(paths.filename_loc_px)
    loc_px.columns = ["pixel_name", "Longitude", "Latitude"]

    location_gauges = pd.read_csv(paths.filename_loc_gauges)
    if "station" not in location_gauges.columns:
        if "codestation" in location_gauges.columns:
            location_gauges["station"] = location_gauges["codestation"].astype(str)
        elif "Station" in location_gauges.columns:
            location_gauges["station"] = location_gauges["Station"].astype(str)
        else:
            raise ValueError("No station id column found (station/codestation/Station).")

    rain_stations = [c for c in rain_hsm.columns if c != "dates"]
    location_gauges = location_gauges[location_gauges["station"].isin(rain_stations)].copy()

    # project lon/lat to meters (Lambert-93)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    # already lambert
    px_x, px_y = [loc_px["Longitude"].to_numpy(), loc_px["Latitude"].to_numpy()]
    loc_px["X_m"] = px_x
    loc_px["Y_m"] = px_y

    # transform in lambert
    g_x, g_y = transformer.transform(location_gauges["Longitude"].to_numpy(), location_gauges["Latitude"].to_numpy())
    location_gauges["X_m"] = g_x
    location_gauges["Y_m"] = g_y

    px_xy = np.column_stack([loc_px["X_m"].to_numpy(), loc_px["Y_m"].to_numpy()])
    g_xy = np.column_stack([location_gauges["X_m"].to_numpy(), location_gauges["Y_m"].to_numpy()])

    # nearest pixel to each gauge
    nearest_idx = np.empty(len(location_gauges), dtype=np.int64)
    for i in range(len(location_gauges)):
        d2 = ((px_xy - g_xy[i]) ** 2).sum(axis=1)
        nearest_idx[i] = int(np.argmin(d2))

    location_gauges["closest_pixel"] = loc_px["pixel_name"].iloc[nearest_idx].to_numpy()
    location_gauges["lon_X"] = loc_px["Longitude"].iloc[nearest_idx].to_numpy()
    location_gauges["lat_X"] = loc_px["Latitude"].iloc[nearest_idx].to_numpy()

    # pixels within radius around the closest pixel
    station_pixels: Dict[str, List[str]] = {}

    for _, row in location_gauges.iterrows():
        st = row["station"]

        cidx = int(np.where(loc_px["pixel_name"].to_numpy() == row["closest_pixel"])[0][0])
        cx, cy = loc_px["X_m"].iloc[cidx], loc_px["Y_m"].iloc[cidx]

        dx = loc_px["X_m"].to_numpy() - cx
        dy = loc_px["Y_m"].to_numpy() - cy
        dist = np.sqrt(dx * dx + dy * dy)

        mask = dist <= radius_m

        pix_df = loc_px.loc[mask, ["pixel_name", "X_m", "Y_m"]].copy()
        pix_df["dist_to_center"] = np.sqrt(
            (pix_df["X_m"] - cx) ** 2 + (pix_df["Y_m"] - cy) ** 2
        )

        pix_df = pix_df.sort_values("dist_to_center")

        station_pixels[st] = pix_df["pixel_name"].astype(str).tolist()

    return location_gauges, loc_px, station_pixels


#Read COMEPHORE only for needed time range using chunks
def read_comephore_filtered(
    filename_com: str,
    tmin: pd.Timestamp,
    tmax: pd.Timestamp,
    chunksize: int = 20000,
) -> pd.DataFrame:
    out = []
    for chunk in pd.read_csv(filename_com, chunksize=chunksize):
        chunk["datetime"] = to_utc(chunk["datetime"])
        sub = chunk[(chunk["datetime"] >= tmin) & (chunk["datetime"] <= tmax)]
        if len(sub):
            out.append(sub)
    if not out:
        raise ValueError("No COMEPHORE data found in requested time window.")
    df = pd.concat(out, ignore_index=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# Build the full table and write CSV
def build_table(
    paths: Paths,
    start: str = "2019-01-01",
    end: str = "2025-01-01",
    radius_m: float = 1500.0,
    n_feat: int = 27,
    n_jobs: int = 1,
    chunksize_com: int = 20000,
) -> None:
    """
    Fxed times: t-1h, t, t+1h (nearest COMEPHORE hour).
    Spatial neighborhood: pixels within radius around closest pixel.
    Features: vec(as.matrix(time x pixels)) in column-major, padded/truncated to n_feat.
    Column names: X_pXX_dt-1h / dt0h / dt+1h.
    """
    # OMSEV
    rain_hsm = load_omsev(paths, start=start, end=end)
    if rain_hsm.empty:
        raise ValueError(f"No OMSEV data in [{start}, {end}).")

    # We need COMEPHORE coverage for t-1h..t+1h
    tmin = rain_hsm["dates"].min() - pd.Timedelta(hours=1)
    tmax = rain_hsm["dates"].max() + pd.Timedelta(hours=1)

    # station -> pixels within radius
    location_gauges, _, station_pixels = build_station_radius_pixels(
        paths=paths,
        rain_hsm=rain_hsm,
        radius_m=radius_m,
    )

    # COMEPHORE
    comephore = read_comephore_filtered(
        paths.filename_com, tmin=tmin, tmax=tmax, chunksize=chunksize_com
    )

    pixel_cols = [c for c in comephore.columns if c != "datetime"]
    pixel_to_col = {p: i for i, p in enumerate(pixel_cols)}

    com_times = comephore["datetime"].to_numpy(dtype="datetime64[ns]")
    com_ns = com_times.astype("int64")  # sorted
    com_mat = comephore[pixel_cols].to_numpy(dtype=np.float32)

    # output columns
    X_cols = make_X_names(n_feat=n_feat)
    out_cols = ["time", "station", "lon_Y", "lat_Y", "lon_X", "lat_X", "Y_obs"] + X_cols

    # station metadata lookup
    st_lonY = dict(zip(location_gauges["station"], location_gauges["Longitude"]))
    st_latY = dict(zip(location_gauges["station"], location_gauges["Latitude"]))
    st_lonX = dict(zip(location_gauges["station"], location_gauges["lon_X"]))
    st_latX = dict(zip(location_gauges["station"], location_gauges["lat_X"]))

    gauge_cols = [c for c in rain_hsm.columns if c != "dates"]

    # per-time processing
    def process_one_time(ti: int) -> List[dict]:
        t_date = rain_hsm.iloc[ti]["dates"]

        # t-1h, t, t+1h
        targets = [
            (t_date - pd.Timedelta(hours=1)).to_datetime64(),
            (t_date).to_datetime64(),
            (t_date + pd.Timedelta(hours=1)).to_datetime64(),
        ]
        targets_ns = [np.datetime64(tt, "ns").astype("int64") for tt in targets]
        idxs = np.array([nearest_time_index_one(tt, com_ns) for tt in targets_ns], dtype=np.int64)

        row = rain_hsm.iloc[ti]
        out_rows = []

        for st in gauge_cols:
            y = row[st]
            if pd.isna(y):
                continue

            pix_list = station_pixels.get(st, [])
            if not pix_list:
                continue

            pix_idx = [pixel_to_col[p] for p in pix_list if p in pixel_to_col]
            if not pix_idx:
                continue

            cube = com_mat[idxs, :][:, pix_idx]

            vec = cube.reshape(-1, order="F").astype(np.float64)

            if vec.size >= n_feat:
                x = vec[:n_feat]
            else:
                x = np.full(n_feat, np.nan, dtype=np.float64)
                x[: vec.size] = vec

            d = {
                "time": pd.Timestamp(t_date).to_pydatetime(),
                "station": st,
                "lon_Y": float(st_lonY[st]),
                "lat_Y": float(st_latY[st]),
                "lon_X": float(st_lonX[st]),
                "lat_X": float(st_latX[st]),
                "Y_obs": float(y),
            }
            for j, col in enumerate(X_cols):
                v = x[j]
                d[col] = float(v) if not np.isnan(v) else np.nan

            out_rows.append(d)

        return out_rows

    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        delayed(process_one_time)(i) for i in range(len(rain_hsm))
    )

    flat = [r for sub in results for r in sub]
    out_df = pd.DataFrame(flat).reindex(columns=out_cols)

    os.makedirs(os.path.dirname(paths.output_file), exist_ok=True)
    out_df.to_csv(paths.output_file, sep=";", index=False)