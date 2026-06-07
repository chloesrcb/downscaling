# %%
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from downscaling.paths import DOWNSCALING_TABLE, IM_FOLDER
from downscaling.data import (
    attach_nearest_comephore_pixels,
    build_fine_grid_radar_predictors_5min,
    find_col,
    make_covariate_sets,
    make_fine_grid_from_gauges,
    prepare_comephore_pixels,
    prepare_modeling_dataframe,
    read_comephore_wide,
)
from downscaling.occurrence import prepare_occurrence_dataframe
from downscaling.plotting import (
    DRY_CMAP,
    PARAM_CMAP,
    RAIN_CMAP,
    configure_plot_style,
    plot_summary_set,
)
from downscaling.prediction import predict_intensity_on_grid, predict_occurrence_on_grid_batched
from downscaling.simulations import (
    add_distribution_indicators,
    summarize_dry_periods,
    summarize_month_maps,
)


# %%
configure_plot_style()

SEED = 2026

GRID_RES_M = 100
GRID_BUFFER_M = 500

MONTH_LABEL = "sep2020"
START_DATE = "2020-09-01"
END_DATE = "2020-10-01"

# Predictions are made at 5-min resolution.
PRED_FREQ = "5min"

# Monte Carlo size for conditional means.
# Quantiles are computed directly from qegpd, so this does not affect q90/q95/q99.
N_SIM_MEAN = 200
N_SIM_DRY = 200

DATA_FOLDER = Path(os.environ.get("DATA_FOLDER", "../../phd_extremes/data/"))

FILE_COMEPHORE = DATA_FOLDER / "comephore/rebuild_clean/comephore_2008_2025_within5km.csv"
FILE_PIXELS = DATA_FOLDER / "comephore/rebuild_clean/coords_pixels_within5km.csv"
FILE_GAUGES = DATA_FOLDER / "omsev/loc_rain_gauges.csv"

OUT_DIR = Path(IM_FOLDER) / f"fine_grid_{GRID_RES_M}m_downscaling_{MONTH_LABEL}_5min"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_PARAMS = {
    "variant": "both",
    "x_set_name": "radar_time_space",
    "widths": (8, 4),
    "lr": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 128,
    "n_ep": 300,
    "sigma_init": 0.58,
    "kappa_init": 0.27,
    "xi_init": 0.20,
    "censor_threshold": 0.3,
    "init_source": "default",
    "kappa_max_nn": 1.0,
    "lambda_kappa": 5.0,
}

MAP_SPECS = [
    {
        "col": "prob_occ_5min_mean",
        "name": "prob_occ_5min_mean",
        "label": "Mean 5-min rainfall occurrence probability",
        "scale": {"mode": "quantile", "qmin": 0.02, "qmax": 0.98},
    },
    {
        "col": "expected_wet_5min_steps",
        "name": "expected_wet_5min_steps",
        "label": "Expected number of wet 5-min steps",
        "scale": {"mode": "quantile", "qmin": 0.02, "qmax": 0.98},
    },
    {
        "col": "rain_monthly_sum_expected",
        "name": "rain_monthly_sum_expected",
        "label": "Expected monthly rainfall accumulation",
        "scale": {"mode": "quantile", "qmin": 0.02, "qmax": 0.98},
    },
    {
        "col": "rain_pos_mean",
        "name": "rain_pos_mean",
        "label": "Mean rainfall | X > 0",
        "scale": {"mode": "positive_quantile", "vmin": 0.0, "qmax": 0.98},
    },
    {
        "col": "rain_pos_q90",
        "name": "rain_pos_q90",
        "label": "Positive rainfall q90 | X > 0",
        "scale": {"mode": "positive_quantile", "vmin": 0.0, "qmax": 0.98},
    },
    {
        "col": "rain_pos_q95",
        "name": "rain_pos_q95",
        "label": "Positive rainfall q95 | X > 0",
        "scale": {"mode": "positive_quantile", "vmin": 0.0, "qmax": 0.98},
    },
    {
        "col": "rain_pos_q99",
        "name": "rain_pos_q99",
        "label": "Positive rainfall q99 | X > 0",
        "scale": {"mode": "positive_quantile", "vmin": 0.0, "qmax": 0.98},
    },
    {
        "col": "log_rain_pos_mean",
        "name": "log_rain_pos_mean",
        "label": "Mean log(1 + X) | X > 0",
        "scale": {"mode": "quantile", "qmin": 0.02, "qmax": 0.98},
    },
    {
        "col": "log_rain_pos_q95",
        "name": "log_rain_pos_q95",
        "label": "q95 of log(1 + X) | X > 0",
        "scale": {"mode": "quantile", "qmin": 0.02, "qmax": 0.98},
    },
    {
        "col": "log_rain_pos_q99",
        "name": "log_rain_pos_q99",
        "label": "q99 of log(1 + X) | X > 0",
        "scale": {"mode": "quantile", "qmin": 0.02, "qmax": 0.98},
    },
    {
        "col": "mean_above_row_q95",
        "name": "mean_above_row_q95",
        "label": "Mean rainfall | X > local 5-min q95",
        "scale": {"mode": "positive_quantile", "vmin": 0.0, "qmax": 0.98},
    },
    {
        "col": "sigma_mean",
        "name": "sigma_mean",
        "label": "Mean predicted sigma",
        "cmap": PARAM_CMAP,
        "scale": {"mode": "quantile", "qmin": 0.02, "qmax": 0.98},
    },
    {
        "col": "kappa_mean",
        "name": "kappa_mean",
        "label": "Mean predicted kappa",
        "cmap": PARAM_CMAP,
        "scale": {"mode": "quantile", "qmin": 0.02, "qmax": 0.98},
    },
    {
        "col": "radar_mean_dt0h",
        "name": "radar_mean_dt0h",
        "label": "Mean hourly radar rainfall at dt0h",
        "scale": {"mode": "positive_quantile", "vmin": 0.0, "qmax": 0.98},
    },
]


DRY_MAP_SPECS = [
    {
        "col": "wet_5min_fraction",
        "name": "wet_5min_fraction",
        "label": "Simulated fraction of wet 5-min steps",
        "cmap": RAIN_CMAP,
        "scale": {"mode": "quantile", "qmin": 0.02, "qmax": 0.98},
    },
    {
        "col": "dry_day_fraction",
        "name": "dry_day_fraction",
        "label": "Simulated fraction of dry days",
        "cmap": DRY_CMAP,
        "scale": {"mode": "quantile", "qmin": 0.02, "qmax": 0.98},
    },
    {
        "col": "n_dry_days_mean",
        "name": "n_dry_days_mean",
        "label": "Mean number of dry days",
        "cmap": DRY_CMAP,
        "scale": {"mode": "quantile", "qmin": 0.02, "qmax": 0.98},
    },
    {
        "col": "max_consecutive_dry_days_mean",
        "name": "max_consecutive_dry_days_mean",
        "label": "Mean maximum consecutive dry days",
        "cmap": DRY_CMAP,
        "scale": {"mode": "quantile", "qmin": 0.02, "qmax": 0.98},
    },
    {
        "col": "max_consecutive_dry_days_q95",
        "name": "max_consecutive_dry_days_q95",
        "label": "q95 maximum consecutive dry days",
        "cmap": DRY_CMAP,
        "scale": {"mode": "quantile", "qmin": 0.02, "qmax": 0.98},
    },
]

#%%
# %%
# Load training data
df_raw = pd.read_csv(DOWNSCALING_TABLE, sep=";")
df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True)

df_model, x_cols27, x_cols_dt0h, x_cols_all = prepare_modeling_dataframe(df_raw)

x_sets = make_covariate_sets(
    x_cols27=x_cols27,
    x_cols_dt0h=x_cols_dt0h,
    x_cols_all=x_cols_all,
)

x_set_name = BEST_PARAMS["x_set_name"]
x_cols_intensity = x_sets[x_set_name]

print("Intensity covariates:")
print(x_cols_intensity)


# %%
# Build fine grid
gauges = pd.read_csv(FILE_GAUGES)
gauges = gauges[~gauges["Station"].isin(["brives", "hydro", "cines"])].copy()

station_col_g = find_col(gauges.columns, ["station", "Station", "site", "name", "gauge", "id"])
lon_g = find_col(gauges.columns, ["lon", "Longitude", "lon_Y"])
lat_g = find_col(gauges.columns, ["lat", "Latitude", "lat_Y"])

gauges = gauges.rename(columns={
    station_col_g: "station",
    lon_g: "lon_Y",
    lat_g: "lat_Y",
})

to_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)

gauges_plot = gauges.copy()
gauges_plot["x_l93"], gauges_plot["y_l93"] = to_l93.transform(
    gauges_plot["lon_Y"].to_numpy(float),
    gauges_plot["lat_Y"].to_numpy(float),
)

fine_grid = make_fine_grid_from_gauges(
    gauges=gauges,
    lon_col="lon_Y",
    lat_col="lat_Y",
    res_m=GRID_RES_M,
    buffer_m=GRID_BUFFER_M,
)

comephore = read_comephore_wide(FILE_COMEPHORE)
pixels = prepare_comephore_pixels(FILE_PIXELS, comephore_cols=comephore.columns)

fine_grid = attach_nearest_comephore_pixels(
    fine_grid=fine_grid,
    pixels=pixels,
    n_pixels=9,
)

fine_grid.to_csv(
    OUT_DIR / f"fine_grid_{GRID_RES_M}m_with_nearest_comephore_pixels.csv",
    index=False,
)

print("Fine grid points:", len(fine_grid))
print("Unique central COMEPHORE pixels:", fine_grid["p01_id"].nunique())


# %%
# Build 5-min prediction table
grid_pred = build_fine_grid_radar_predictors_5min(
    fine_grid=fine_grid,
    comephore=comephore,
    start_date=START_DATE,
    end_date=END_DATE,
    pred_freq=PRED_FREQ,
)

missing_int = [c for c in x_cols_intensity if c not in grid_pred.columns]
if missing_int:
    raise ValueError(f"Missing intensity covariates in grid table:\n{missing_int}")

print(grid_pred.shape)
print("Unique grid ids:", grid_pred["grid_id"].nunique())
print("Unique 5-min times:", grid_pred["time"].nunique())


# %%
# Predict intensity parameters
grid_pred = predict_intensity_on_grid(
    df_model=df_model,
    grid_pred=grid_pred,
    x_sets=x_sets,
    best_params=BEST_PARAMS,
    seed=SEED,
)

print(grid_pred[["sigma", "kappa", "xi"]].describe())


# %%
# Predict occurrence probabilities at 5-min scale
df_occ, _, x_cols_occ = prepare_occurrence_dataframe(
    df_raw,
    use_time=True,
    use_spatial=True,
    use_summaries=True,
    use_cube=True,
)

missing_occ = [c for c in x_cols_occ if c not in grid_pred.columns]
if missing_occ:
    raise ValueError(f"Missing occurrence covariates in grid table:\n{missing_occ}")

grid_pred["p_occ_hat"] = predict_occurrence_on_grid_batched(
    df_occ_train=df_occ,
    grid_pred=grid_pred,
    x_cols=x_cols_occ,
    batch_size_pred=100_000,
    seed=SEED,
).clip(0, 1)

print(grid_pred["p_occ_hat"].describe())


# %%
# Add exact quantiles and Monte Carlo conditional means
grid_pred = add_distribution_indicators(
    grid_pred=grid_pred,
    n_sim_mean=N_SIM_MEAN,
    batch_size=100_000,
    seed=SEED,
)

grid_pred.to_csv(OUT_DIR / f"fine_grid_predictions_{MONTH_LABEL}_5min.csv", index=False)


# %%
# Summaries over all 5-min time steps
summary_all = summarize_month_maps(
    grid_pred=grid_pred,
    selection_label="all_5min_steps",
    month_label=MONTH_LABEL,
    mask=None,
)

summary_all.to_csv(
    OUT_DIR / f"summary_{MONTH_LABEL}_all_5min_steps.csv",
    index=False,
)

plot_summary_set(
    summary=summary_all,
    prefix=f"{MONTH_LABEL}_all_5min_steps",
    map_specs=MAP_SPECS,
    out_dir=OUT_DIR,
    gauges_plot=gauges_plot,
    grid_res_m=GRID_RES_M,
)


# %%
# summaries restricted to radar-wet 5-min steps
mask_radar_wet = grid_pred["radar_mean_dt0h"] > 0

summary_radar_wet = summarize_month_maps(
    grid_pred=grid_pred,
    selection_label="radar_wet_5min_steps",
    month_label=MONTH_LABEL,
    mask=mask_radar_wet,
)

summary_radar_wet.to_csv(
    OUT_DIR / f"summary_{MONTH_LABEL}_radar_wet_5min_steps.csv",
    index=False,
)

plot_summary_set(
    summary=summary_radar_wet,
    prefix=f"{MONTH_LABEL}_radar_wet_5min_steps",
    map_specs=MAP_SPECS,
    out_dir=OUT_DIR,
    gauges_plot=gauges_plot,
    grid_res_m=GRID_RES_M,
)


# %%
# Dry-period summaries from simulated occurrence sequences
summary_dry = summarize_dry_periods(
    grid_pred=grid_pred,
    n_sim_dry=N_SIM_DRY,
    seed=SEED,
    month_label=MONTH_LABEL,
)

summary_dry.to_csv(
    OUT_DIR / f"summary_{MONTH_LABEL}_dry_periods.csv",
    index=False,
)

plot_summary_set(
    summary=summary_dry,
    prefix=f"{MONTH_LABEL}_dry_periods",
    map_specs=DRY_MAP_SPECS,
    out_dir=OUT_DIR,
    gauges_plot=gauges_plot,
    grid_res_m=GRID_RES_M,
)

print("Done. Outputs in:", OUT_DIR)
