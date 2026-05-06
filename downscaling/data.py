import re
import numpy as np
import pandas as pd

from downscaling.config import TIME_COLS, SPATIAL_COLS


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