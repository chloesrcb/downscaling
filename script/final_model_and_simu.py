# %%
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from downscaling.settings import DOWNSCALING_TABLE, IM_FOLDER

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
from downscaling.plotting import configure_plot_style

configure_plot_style()

SEED = 2026

GRID_RES_M = 100
GRID_BUFFER_M = 500

MONTH_LABEL = "sep2022"
START_DATE = "2022-09-01"
END_DATE = "2022-10-01"

# Predictions are made at 5-min resolution.
PRED_FREQ = "5min"

# Monte Carlo size for conditional means.
# Quantiles are computed directly from qegpd, so this does not affect q90/q95/q99.
N_SIM_MEAN = 100
N_SIM_DRY = 100

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
    "xi_init": 0.18,
    "censor_threshold": 0.3,
    "init_source": "default",
    "kappa_max_nn": 1.0,
    "lambda_kappa": 5.0,
}

MAP_SPECS = [
    {
    "col": "prob_occ_5min_mean",
    "name": "prob_occ_5min_mean",
    "label": "Occurrence probability (%)",
    "display_factor": 100.0,
    "cbar_format": "%.2f",
    "scale": {"mode": "quantile", "qmin": 0.00, "qmax": 1.00},
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
from downscaling.prediction import predict_intensity_on_grid
# Predict intensity parameters
grid_pred = predict_intensity_on_grid(
    df_model=df_model,
    grid_pred=grid_pred,
    x_sets=x_sets,
    best_params=BEST_PARAMS,
    seed=SEED,
)

print(grid_pred[["sigma", "kappa", "xi"]].describe())

#%%
# number of predictors
print("Number of intensity predictors:", len(x_cols_intensity))



# %%
from downscaling.prediction import predict_occurrence_on_grid_batched
# Predict occurrence probabilities at 5-min scale
df_occ, _, x_cols_occ = prepare_occurrence_dataframe(
    df_raw,
    use_time=True,
    use_spatial=True,
    use_summaries=True,
    use_cube=False,
    remove_incoherent=False,
    summary_scale="raw"
)

# Add occurrence-specific covariates to the grid table
grid_pred["radar_any"] = (grid_pred["radar_sum"] > 0).astype(int)

# adapt this name if your central pixel dt0h has another name
center_col = "X_p01_dt0h"
grid_pred["radar_center"] = (grid_pred[center_col] > 0).astype(int)

missing_occ = [c for c in x_cols_occ if c not in grid_pred.columns]
if missing_occ:
    raise ValueError(f"Missing occurrence covariates in grid table:\n{missing_occ}")

#%%
grid_pred["p_occ_hat"] = predict_occurrence_on_grid_batched(
    df_occ_train=df_occ,
    grid_pred=grid_pred,
    x_cols=x_cols_occ,
    batch_size_pred=100_000,
    seed=SEED,
).clip(0, 1)

print(grid_pred["p_occ_hat"].describe())


# %%
from downscaling.simulations import add_distribution_indicators
# Add exact quantiles and Monte Carlo conditional means
grid_pred = add_distribution_indicators(
    grid_pred=grid_pred,
    n_sim_mean=N_SIM_MEAN,
    batch_size=100_000,
    seed=SEED,
)

grid_pred.to_csv(OUT_DIR / f"fine_grid_predictions_{MONTH_LABEL}_5min.csv", index=False)


# %%
from downscaling.simulations import summarize_month_maps
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
#%%

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


# %%
# Dry-period summaries from simulated occurrence sequences
from downscaling.simulations import summarize_dry_periods
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

#%%
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
PALETTE = {
    "Stationary EGPD": "#8B0000",  # darkred
    "Stationary": "#8B0000",
    "GLM": "#F28E2B",              # orange
    "GAM": "#D45087",              # rose
    "NN": "#008080",               # teal
    "Observed": "#8B0000",
    "Simulated": "#008080",
    "CRPS": "#D45087",
    "twCRPS": "#008080",
    "CRPSS": "#D45087",
    "twCRPSS": "#008080",
    "darkred": "#8B0000",
}

SCORE_LABELS = {
    "twcrps_sum": "Tail-weighted CRPS",
    "twcrps_mean": "Tail-weighted CRPS mean",
    "crps_mean": "Mean CRPS",
    "smad": "sMAD",
    "pit_cvm": "PIT CvM",
    "kappa_q99": r"99th percentile of $\kappa$",
    "prop_kappa_gt_2": r"Proportion of $\kappa > 2$",
    "crps_skill": "SS CRPS",
    "twcrps_skill": "SS twCRPS",
}

import matplotlib.colors as mcolors
RAIN_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "rain_blue",
    ["#F7FBFF", "#DEEBF7", "#9ECAE1", "#4292C6", "#08519C"],
    N=256,
)

DRY_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "dry_orange",
    ["#FFF7EC", "#FDD49E", "#FDBB84", "#EF6548", "#990000"],
    N=256,
)

PARAM_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "param_purple",
    ["#F7FCFD", "#E0ECF4", "#BFD3E6", "#9EBCDA", "#8C96C6", "#88419D"],
    N=256,
)
# Choose summary manually
summary_plot = summary_all.copy()
# summary_plot = summary_radar_wet.copy()


from matplotlib.ticker import FormatStrFormatter

def plot_one_map(
    summary_plot,
    value_col,
    label,
    filename,
    vmin=None,
    vmax=None,
    cmap=RAIN_CMAP,
    display_factor=1.0,
    cbar_format="%.2f",
    gauges_plot=None,
    grid_res_m=GRID_RES_M,
):
    dat = summary_plot[["x_l93", "y_l93", value_col]].copy()
    dat = dat.replace([np.inf, -np.inf], np.nan)
    dat = dat.dropna(subset=["x_l93", "y_l93", value_col])

    pivot = (
        dat.pivot_table(index="y_l93", columns="x_l93", values=value_col,
                        aggfunc="mean", dropna=False)
        .sort_index()
        .sort_index(axis=1)
    )

    raw_values = pivot.values.astype(float) * display_factor
    finite_values = raw_values[np.isfinite(raw_values)]

    data_min = np.nanmin(finite_values)
    data_max = np.nanmax(finite_values)

    print(f"\n{value_col}")
    print(dat[value_col].mul(display_factor).describe())
    print(f"min = {data_min:.4f}, max = {data_max:.4f}")

    if vmin is None:
        vmin = data_min
    if vmax is None:
        vmax = data_max

    print(f"plot scale: vmin = {vmin:.4f}, vmax = {vmax:.4f}")

    x = np.sort(pivot.columns.to_numpy(float))
    y = np.sort(pivot.index.to_numpy(float))

    x_edges = np.r_[x[0] - grid_res_m / 2, (x[:-1] + x[1:]) / 2, x[-1] + grid_res_m / 2]
    y_edges = np.r_[y[0] - grid_res_m / 2, (y[:-1] + y[1:]) / 2, y[-1] + grid_res_m / 2]

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.pcolormesh(
        x_edges, y_edges,
        np.ma.masked_invalid(raw_values),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(cbar_format))

    if gauges_plot is not None:
        ax.scatter(
            gauges_plot["x_l93"], gauges_plot["y_l93"],
            s=35, c=PALETTE["darkred"],
            edgecolor="white", linewidth=0.5, zorder=5,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
#%%
plot_one_map(
    summary_plot=summary_all,
    value_col="prob_occ_5min_mean",
    label="Mean occurrence probability (%)",
    filename=OUT_DIR / f"map_{MONTH_LABEL}_prob_occ_percent.png",
    vmin=0.8,
    vmax=2.5,
    display_factor=100.0,
    cmap=RAIN_CMAP,
    cbar_format="%.2f",
    gauges_plot=gauges_plot,
)
#%%
plot_one_map(
    summary_plot=summary_all,
    value_col="rain_pos_mean",
    label="Mean rainfall | X > 0 (mm / 5 min)",
    filename=OUT_DIR / f"map_{MONTH_LABEL}_rain_pos_mean.png",
    vmin=0.05,
    vmax=0.15,
    cmap=RAIN_CMAP,
    gauges_plot=gauges_plot,
)

#%%
plot_one_map(
    summary_plot=summary_all,
    value_col="mean_above_row_q95",
    label="Mean rainfall | X > local q95 (mm / 5 min)",
    filename=OUT_DIR / f"map_{MONTH_LABEL}_mean_above_q95.png",
    vmin=0.3,
    vmax=0.7,
    cmap=RAIN_CMAP,
    gauges_plot=gauges_plot,
)

#%%
plot_one_map(
    summary_plot=summary_all,
    value_col="rain_monthly_sum_expected",
    label="Expected monthly rainfall accumulation (mm)",
    filename=OUT_DIR / f"map_{MONTH_LABEL}_rain_monthly_sum_expected.png",
    vmin=50,
    vmax=420,
    cmap=RAIN_CMAP,
    gauges_plot=gauges_plot,
)

#%%
plot_one_map(
    summary_plot=summary_all,
    value_col="log_rain_pos_q95",
    label="95th percentile (mm / 5 min, log scale)",
    filename=OUT_DIR / f"map_{MONTH_LABEL}_log_rain_pos_q95.png",
    vmin=0.15,
    vmax=0.35,
    cmap=RAIN_CMAP,
    gauges_plot=gauges_plot,
)

#%%
plot_one_map(
    summary_plot=summary_all,
    value_col="log_rain_pos_q99",
    label="99th percentile (mm / 5 min, log scale)",
    filename=OUT_DIR / f"map_{MONTH_LABEL}_log_rain_pos_q99.png",
    vmin=0.35,


    vmax=0.6,
    cmap=RAIN_CMAP,
    gauges_plot=gauges_plot,
)

#%%
plot_one_map(
    summary_plot=summary_all,
    value_col="rain_pos_q95",
    label="95th percentile (mm / 5 min)",
    filename=OUT_DIR / f"map_{MONTH_LABEL}_rain_pos_q95.png",
    vmin=0.2,

    vmax=0.45,
    cmap=RAIN_CMAP,
    gauges_plot=gauges_plot,
)

#%%
plot_one_map(
    summary_plot=summary_all,
    value_col="rain_pos_q99",
    label="99th percentile (mm / 5 min)",
    filename=OUT_DIR / f"map_{MONTH_LABEL}_rain_pos_q99.png",
    # vmin=0.35,

    # vmax=0.6,
    cmap=RAIN_CMAP,
    gauges_plot=gauges_plot,
)

#%%
plot_one_map(
    summary_plot=summary_dry,
    value_col="dry_day_fraction",
    label="Dry day fraction",
    filename=OUT_DIR / f"map_{MONTH_LABEL}_dry_day_fraction.png",
    # vmin=0.0,
    # vmax=1.0,
    cmap=DRY_CMAP,
    gauges_plot=gauges_plot,
)

#%%
obs_month = df_raw.copy()
obs_month["time"] = pd.to_datetime(obs_month["time"], utc=True)

obs_month = obs_month[
    (obs_month["time"] >= START_DATE) &
    (obs_month["time"] < END_DATE) &
    obs_month["Y_obs"].notna()
].copy()

obs_summary = (
    obs_month
    .groupby("station")
    .agg(
        obs_wet_5min_fraction=("Y_obs", lambda x: np.mean(x > 0)),
        obs_monthly_sum=("Y_obs", "sum"),
        obs_pos_mean=("Y_obs", lambda x: x[x > 0].mean()),
        obs_pos_q95=("Y_obs", lambda x: x[x > 0].quantile(0.95) if np.any(x > 0) else np.nan),
        n_obs=("Y_obs", "size"),
    )
    .reset_index()
)

#%%
obs_summary = obs_summary.merge(
    gauges_plot[["station", "x_l93", "y_l93"]],
    on="station",
    how="left"
)

#%%
from scipy.spatial import cKDTree

grid_xy = summary_all[["x_l93", "y_l93"]].to_numpy(float)
sta_xy = obs_summary[["x_l93", "y_l93"]].to_numpy(float)

tree = cKDTree(grid_xy)
dist, idx = tree.query(sta_xy, k=1)

grid_at_station = summary_all.iloc[idx].reset_index(drop=True).copy()
grid_at_station = grid_at_station.add_prefix("sim_")

comp = pd.concat(
    [
        obs_summary.reset_index(drop=True),
        grid_at_station,
    ],
    axis=1,
)

comp["dist_to_grid_m"] = dist

#%%
comp["bias_monthly_sum"] = comp["sim_rain_monthly_sum_expected"] - comp["obs_monthly_sum"]
comp["bias_wet_fraction"] = comp["sim_prob_occ_5min_mean"] - comp["obs_wet_5min_fraction"]
comp["bias_pos_mean"] = comp["sim_rain_pos_mean"] - comp["obs_pos_mean"]

print(comp[
    [
        "station",
        "obs_monthly_sum",
        "sim_rain_monthly_sum_expected",
        "bias_monthly_sum",
        "obs_wet_5min_fraction",
        "sim_prob_occ_5min_mean",
        "bias_wet_fraction",
        "obs_pos_mean",
        "sim_rain_pos_mean",
        "bias_pos_mean",
        "dist_to_grid_m",
    ]
])

#%%
plot_one_map(
    summary_plot=summary_all,
    value_col="rain_monthly_sum_expected",
    label="Expected monthly rainfall accumulation (mm)",
    filename=OUT_DIR / f"map_{MONTH_LABEL}_monthly_sum_with_station_bias.png",
    vmin=None,
    vmax=None,
    cmap=RAIN_CMAP,
    gauges_plot=None,
)

#%%
fig, ax = plt.subplots(figsize=(8, 7))

# fond simulé
dat = summary_all[["x_l93", "y_l93", "rain_monthly_sum_expected"]].dropna()
pivot = dat.pivot_table(
    index="y_l93",
    columns="x_l93",
    values="rain_monthly_sum_expected",
    aggfunc="mean"
).sort_index().sort_index(axis=1)

raw_values = pivot.values.astype(float)

x = np.sort(pivot.columns.to_numpy(float))
y = np.sort(pivot.index.to_numpy(float))

x_edges = np.r_[x[0] - GRID_RES_M / 2, (x[:-1] + x[1:]) / 2, x[-1] + GRID_RES_M / 2]
y_edges = np.r_[y[0] - GRID_RES_M / 2, (y[:-1] + y[1:]) / 2, y[-1] + GRID_RES_M / 2]

im = ax.pcolormesh(
    x_edges,
    y_edges,
    raw_values,
    cmap=RAIN_CMAP,
    shading="flat",
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Expected monthly rainfall accumulation (mm)")

# stations colorées par biais
sc = ax.scatter(
    comp["x_l93"],
    comp["y_l93"],
    c=comp["bias_monthly_sum"],
    s=80,
    cmap="coolwarm",
    edgecolor="black",
    linewidth=0.7,
    zorder=5,
)

cbar2 = fig.colorbar(sc, ax=ax)
cbar2.set_label("Bias at stations: simulated - observed (mm)")

ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal", adjustable="box")

fig.tight_layout()
fig.savefig(OUT_DIR / f"map_{MONTH_LABEL}_monthly_sum_station_bias.png", dpi=300, bbox_inches="tight")
plt.show()

#%%
fig, ax = plt.subplots(figsize=(5, 5))

ax.scatter(
    comp["obs_monthly_sum"],
    comp["sim_rain_monthly_sum_expected"],
    s=50,
    edgecolor="black",
)

lim_min = min(comp["obs_monthly_sum"].min(), comp["sim_rain_monthly_sum_expected"].min())
lim_max = max(comp["obs_monthly_sum"].max(), comp["sim_rain_monthly_sum_expected"].max())

ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1)

ax.set_xlabel("Observed monthly accumulation (mm)")
ax.set_ylabel("Simulated expected monthly accumulation (mm)")
ax.set_aspect("equal", adjustable="box")

fig.tight_layout()
fig.savefig(OUT_DIR / f"scatter_{MONTH_LABEL}_obs_vs_sim_monthly_sum.png", dpi=300, bbox_inches="tight")
plt.show()

#%%
comp["obs_minus_sim"] = comp["obs_monthly_sum"] - comp["sim_rain_monthly_sum_expected"]

print(comp[[
    "station",
    "obs_monthly_sum",
    "sim_rain_monthly_sum_expected",
    "bias_monthly_sum",
    "obs_wet_5min_fraction",
    "sim_prob_occ_5min_mean",
    "obs_pos_mean",
    "sim_rain_pos_mean",
    "dist_to_grid_m",
]].sort_values("bias_monthly_sum"))

#%%
grid_pred["rain_expected"] = grid_pred["p_occ_hat"] * grid_pred["rain_pos_mean"]
grid_pred["rain_monthly_sum_expected_check"] = grid_pred["rain_expected"] * (31 * 24 * 60 / 5)