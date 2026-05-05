# %%
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
DATA_FOLDER = os.environ.get("DATA_FOLDER", "../phd_extremes/data/")
DOWNSCALING_TABLE = os.path.join(
    DATA_FOLDER,
    "downscaling/downscaling_table_named_2019_2024.csv"
)
IM_FOLDER = "../phd_extremes/thesis/resources/images/downscaling/"
OUT_DIR = os.path.join(
    IM_FOLDER,
    "analysis_figures"
)

FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# %%
TIME_COLS = ["tod_sin", "tod_cos", "doy_sin", "doy_cos", "month_sin", "month_cos"]
SPATIAL_COLS = ["lat_Y", "lon_Y", "lat_X", "lon_X"]

BUCKET_RESOLUTION = 0.2153
RAIN_THRESHOLD_POSITIVE = 0.0

# %%
def savefig(fig, name: str, dpi: int = 300):
    """Save one figure as PNG and PDF."""
    png_path = os.path.join(FIG_DIR, f"{name}.png")
    pdf_path = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def pretty_covariate_name(name: str) -> str:
    """Convert technical covariate names into readable labels for plots."""
    m = re.match(r"^X_p(\d{2})_dt(-1h|0h|\+1h)$", name)
    if m is not None:
        pixel = int(m.group(1))
        lag = m.group(2)

        if lag == "0h":
            lag_label = "t"
        elif lag == "-1h":
            lag_label = "t - 1 h"
        elif lag == "+1h":
            lag_label = "t + 1 h"
        else:
            lag_label = lag

        return f"Radar grid cell {pixel}, {lag_label}"

    replacements = {
        "radar_max": "Radar maximum",
        "radar_mean": "Radar mean",
        "radar_sum": "Radar sum",
        "tod_sin": "Time of day, sine",
        "tod_cos": "Time of day, cosine",
        "doy_sin": "Day of year, sine",
        "doy_cos": "Day of year, cosine",
        "month_sin": "Month, sine",
        "month_cos": "Month, cosine",
        "lat_Y": "Gauge latitude",
        "lon_Y": "Gauge longitude",
        "lat_X": "Radar cell latitude",
        "lon_X": "Radar cell longitude",
        "Y_obs": r"Gauge rainfall $Y_{obs}$",
    }

    return replacements.get(name, name)


def add_grid(ax, axis: str = "both"):
    ax.grid(True, axis=axis, alpha=0.3)


# %%
def get_x_cols27_downscaling(df: pd.DataFrame) -> list[str]:
    pat = re.compile(r"^X_p(\d{2})_dt(-1h|0h|\+1h)$")
    cols = [c for c in df.columns if pat.match(c)]
    order_dt = {"-1h": 0, "0h": 1, "+1h": 2}

    def key(c):
        m = pat.match(c)
        return (int(m.group(1)), order_dt[m.group(2)])

    return sorted(cols, key=key)

def prepare_analysis_dataframe(df_raw: pd.DataFrame):
    """
    Prepare two datasets:
    - df_all: all rows with non-missing Y_obs, including dry cases;
    - df_pos: positive gauge rainfall with radar correspondence, for intensity analysis.

    Important:
    occurrence is computed before filtering positive intensities.
    """
    df = df_raw.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)

    x_cols27 = get_x_cols27_downscaling(df)
    if len(x_cols27) == 0:
        raise ValueError("No radar columns matching X_pXX_dt... were found.")

    df = df.loc[df["Y_obs"].notna()].copy()

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

    # ------------------------------------------------------------
    # Radar summaries: all 27 covariates = 9 pixels x 3 temporal lags
    # ------------------------------------------------------------
    df[x_cols27] = df[x_cols27].apply(pd.to_numeric, errors="coerce")
    X_block_all = df[x_cols27].to_numpy(dtype=float)

    df["radar_max"] = np.nanmax(X_block_all, axis=1)
    df["radar_mean"] = np.nanmean(X_block_all, axis=1)
    df["radar_sum"] = np.nansum(X_block_all, axis=1)

    # ------------------------------------------------------------
    # Radar summaries at current hour only: dt0h
    # ------------------------------------------------------------
    x_cols_dt0h = [c for c in x_cols27 if c.endswith("dt0h")]
    if len(x_cols_dt0h) == 0:
        raise ValueError("No current-hour radar columns ending with dt0h were found.")

    X_block_dt0h = df[x_cols_dt0h].to_numpy(dtype=float)

    df["radar_max_dt0h"] = np.nanmax(X_block_dt0h, axis=1)
    df["radar_mean_dt0h"] = np.nanmean(X_block_dt0h, axis=1)
    df["radar_sum_dt0h"] = np.nansum(X_block_dt0h, axis=1)

    # ------------------------------------------------------------
    # Central/current pixel.
    # If X_p01_dt0h is not the correct central pixel in your data,
    # change this variable.
    # ------------------------------------------------------------
    central_col = "X_p01_dt0h"

    if central_col in df.columns:
        df["radar_central_dt0h"] = df[central_col].astype(float)
    else:
        print(f"Warning: {central_col} not found. Using radar_max_dt0h instead.")
        df["radar_central_dt0h"] = df["radar_max_dt0h"]

    # ------------------------------------------------------------
    # Occurrence definitions
    # ------------------------------------------------------------
    df["gauge_occurrence"] = (df["Y_obs"] > RAIN_THRESHOLD_POSITIVE).astype(int)

    # Very broad definitions
    df["radar_occurrence_sum_all"] = (df["radar_sum"] > 0).astype(int)
    df["radar_occurrence_max_all"] = (df["radar_max"] > 0).astype(int)

    # Current-hour spatial definitions
    df["radar_occurrence_sum_dt0h"] = (df["radar_sum_dt0h"] > 0).astype(int)
    df["radar_occurrence_max_dt0h"] = (df["radar_max_dt0h"] > 0).astype(int)

    # Most local definition
    df["radar_occurrence_central_dt0h"] = (df["radar_central_dt0h"] > 0).astype(int)

    # Definitions using a small physical threshold
    df["radar_occurrence_max_all_bucket"] = (df["radar_max"] >= BUCKET_RESOLUTION).astype(int)
    df["radar_occurrence_max_dt0h_bucket"] = (df["radar_max_dt0h"] >= BUCKET_RESOLUTION).astype(int)
    df["radar_occurrence_central_dt0h_bucket"] = (
        df["radar_central_dt0h"] >= BUCKET_RESOLUTION
    ).astype(int)

    # ------------------------------------------------------------
    # Main definition used in figures/tables.
    #
    # Recommended for occurrence analysis:
    # current-hour local radar value rather than 27-covariate sum.
    # ------------------------------------------------------------
    df["radar_occurrence"] = df["radar_occurrence_central_dt0h"]

    # Joint occurrence used to define the positive-intensity dataset
    df["corres"] = (
        (df["gauge_occurrence"] == 1)
        & (df["radar_occurrence"] == 1)
    ).astype(int)

    keep_cols = [
        "time", "station", "Y_obs", "year", "month", "hour", "minute",

        "gauge_occurrence",
        "radar_occurrence",
        "corres",

        "radar_occurrence_sum_all",
        "radar_occurrence_max_all",
        "radar_occurrence_sum_dt0h",
        "radar_occurrence_max_dt0h",
        "radar_occurrence_central_dt0h",
        "radar_occurrence_max_all_bucket",
        "radar_occurrence_max_dt0h_bucket",
        "radar_occurrence_central_dt0h_bucket",

        *TIME_COLS,
        *SPATIAL_COLS,

        "radar_max", "radar_mean", "radar_sum",
        "radar_max_dt0h", "radar_mean_dt0h", "radar_sum_dt0h",
        "radar_central_dt0h",

        *x_cols27,
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    df_all = df.loc[:, keep_cols].copy()

    df_pos = df_all.loc[
        (df_all["gauge_occurrence"] == 1)
        & (df_all["radar_occurrence"] == 1)
    ].copy()

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
    x_cols_all = [c for c in x_cols_all if c in df_all.columns]

    return df_all, df_pos, x_cols27, x_cols_all


# %%
# Summary tables

def save_summary_tables(df_all: pd.DataFrame, df_pos: pd.DataFrame, x_cols_all: list[str]):
    summary_rows = []

    y_all = df_all["Y_obs"].to_numpy(float)
    y_pos = df_pos["Y_obs"].to_numpy(float)

    summary_rows.append({"quantity": "n_all_nonmissing_y", "value": len(df_all)})
    summary_rows.append({"quantity": "n_positive_with_radar", "value": len(df_pos)})
    summary_rows.append({"quantity": "proportion_y_zero", "value": float(np.mean(y_all == 0))})
    summary_rows.append({"quantity": "proportion_y_positive", "value": float(np.mean(y_all > 0))})
    summary_rows.append({"quantity": "proportion_radar_positive", "value": float(np.mean(df_all["radar_occurrence"] == 1))})
    summary_rows.append({"quantity": "proportion_gauge_and_radar_positive", "value": float(np.mean(df_all["corres"] == 1))})

    for q in [0.50, 0.75, 0.90, 0.95, 0.975, 0.99]:
        summary_rows.append({"quantity": f"positive_y_q{q}", "value": float(np.quantile(y_pos, q))})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(TAB_DIR, "summary_basic.csv"), index=False)

    # Occurrence contingency table.
    occurrence = pd.crosstab(
        df_all["gauge_occurrence"],
        df_all["radar_occurrence"],
        rownames=["gauge_positive"],
        colnames=["radar_positive"],
        normalize=False,
    )
    occurrence.to_csv(os.path.join(TAB_DIR, "occurrence_contingency_counts.csv"))

    occurrence_prop = pd.crosstab(
        df_all["gauge_occurrence"],
        df_all["radar_occurrence"],
        rownames=["gauge_positive"],
        colnames=["radar_positive"],
        normalize="all",
    )
    occurrence_prop.to_csv(os.path.join(TAB_DIR, "occurrence_contingency_proportions.csv"))

    # Correlations with positive rainfall intensity.
    corrs = (
        df_pos[["Y_obs"] + x_cols_all]
        .corr(numeric_only=True)["Y_obs"]
        .drop("Y_obs")
        .sort_values(ascending=False)
    )
    corrs_df = corrs.reset_index()
    corrs_df.columns = ["covariate", "pearson_corr_with_positive_y"]
    corrs_df["pretty_name"] = corrs_df["covariate"].map(pretty_covariate_name)
    corrs_df.to_csv(os.path.join(TAB_DIR, "correlations_positive_intensity.csv"), index=False)

    # Occurrence probabilities by month, hour, and station.
    monthly_occ = (
        df_all.groupby("month")
        .agg(
            n=("Y_obs", "size"),
            gauge_occurrence_rate=("gauge_occurrence", "mean"),
            radar_occurrence_rate=("radar_occurrence", "mean"),
            joint_occurrence_rate=("corres", "mean"),
            positive_mean=("Y_obs", lambda x: np.mean(x[x > 0]) if np.any(x > 0) else np.nan),
            positive_q95=("Y_obs", lambda x: np.quantile(x[x > 0], 0.95) if np.sum(x > 0) > 5 else np.nan),
        )
        .reset_index()
    )
    monthly_occ.to_csv(os.path.join(TAB_DIR, "monthly_occurrence_and_intensity.csv"), index=False)

    hourly_occ = (
        df_all.groupby("hour")
        .agg(
            n=("Y_obs", "size"),
            gauge_occurrence_rate=("gauge_occurrence", "mean"),
            radar_occurrence_rate=("radar_occurrence", "mean"),
            joint_occurrence_rate=("corres", "mean"),
            positive_mean=("Y_obs", lambda x: np.mean(x[x > 0]) if np.any(x > 0) else np.nan),
            positive_q95=("Y_obs", lambda x: np.quantile(x[x > 0], 0.95) if np.sum(x > 0) > 5 else np.nan),
        )
        .reset_index()
    )
    hourly_occ.to_csv(os.path.join(TAB_DIR, "hourly_occurrence_and_intensity.csv"), index=False)

    station_occ = (
        df_all.groupby("station")
        .agg(
            n=("Y_obs", "size"),
            lat_Y=("lat_Y", "first"),
            lon_Y=("lon_Y", "first"),
            gauge_occurrence_rate=("gauge_occurrence", "mean"),
            radar_occurrence_rate=("radar_occurrence", "mean"),
            joint_occurrence_rate=("corres", "mean"),
            positive_mean=("Y_obs", lambda x: np.mean(x[x > 0]) if np.any(x > 0) else np.nan),
            positive_q95=("Y_obs", lambda x: np.quantile(x[x > 0], 0.95) if np.sum(x > 0) > 5 else np.nan),
        )
        .reset_index()
    )
    station_occ.to_csv(os.path.join(TAB_DIR, "station_occurrence_and_intensity.csv"), index=False)

    print("Saved summary tables in:", TAB_DIR)
    return summary, corrs_df, monthly_occ, hourly_occ, station_occ

def save_occurrence_definition_comparison(df_all: pd.DataFrame):
    """
    Compare several radar occurrence definitions on the raw, unfiltered data.
    """
    rows = []

    radar_defs = {
        "all_27_sum_gt_0": "radar_occurrence_sum_all",
        "all_27_max_gt_0": "radar_occurrence_max_all",
        "dt0h_9_pixels_sum_gt_0": "radar_occurrence_sum_dt0h",
        "dt0h_9_pixels_max_gt_0": "radar_occurrence_max_dt0h",
        "central_dt0h_gt_0": "radar_occurrence_central_dt0h",
        "all_27_max_ge_bucket": "radar_occurrence_max_all_bucket",
        "dt0h_9_pixels_max_ge_bucket": "radar_occurrence_max_dt0h_bucket",
        "central_dt0h_ge_bucket": "radar_occurrence_central_dt0h_bucket",
    }

    for name, col in radar_defs.items():
        if col not in df_all.columns:
            print(f"Skipping {name}: column {col} not found.")
            continue

        gauge = df_all["gauge_occurrence"].astype(bool)
        radar = df_all[col].astype(bool)

        n00 = int((~gauge & ~radar).sum())
        n01 = int((~gauge & radar).sum())
        n10 = int((gauge & ~radar).sum())
        n11 = int((gauge & radar).sum())
        n = len(df_all)

        p00 = n00 / n
        p01 = n01 / n
        p10 = n10 / n
        p11 = n11 / n

        p_gauge_wet = float(gauge.mean())
        p_radar_wet = float(radar.mean())

        p_gauge_wet_given_radar_wet = (
            n11 / (n01 + n11) if (n01 + n11) > 0 else np.nan
        )
        p_radar_wet_given_gauge_wet = (
            n11 / (n10 + n11) if (n10 + n11) > 0 else np.nan
        )
        p_false_alarm_given_radar_wet = (
            n01 / (n01 + n11) if (n01 + n11) > 0 else np.nan
        )
        p_missed_gauge_wet = (
            n10 / (n10 + n11) if (n10 + n11) > 0 else np.nan
        )

        rows.append({
            "radar_definition": name,
            "n": n,

            "n_gauge_dry_radar_dry": n00,
            "n_gauge_dry_radar_wet": n01,
            "n_gauge_wet_radar_dry": n10,
            "n_gauge_wet_radar_wet": n11,

            "p_gauge_dry_radar_dry": p00,
            "p_gauge_dry_radar_wet": p01,
            "p_gauge_wet_radar_dry": p10,
            "p_gauge_wet_radar_wet": p11,

            "p_gauge_wet": p_gauge_wet,
            "p_radar_wet": p_radar_wet,
            "p_gauge_wet_given_radar_wet": p_gauge_wet_given_radar_wet,
            "p_radar_wet_given_gauge_wet": p_radar_wet_given_gauge_wet,
            "p_false_alarm_given_radar_wet": p_false_alarm_given_radar_wet,
            "p_missed_gauge_wet": p_missed_gauge_wet,
        })

    out = pd.DataFrame(rows)

    out_path = os.path.join(TAB_DIR, "occurrence_definition_comparison.csv")
    out.to_csv(out_path, index=False)

    print("\nSaved radar occurrence definition comparison:")
    print(out_path)

    return out

def save_occurrence_latex_table(df_all: pd.DataFrame):
    """
    Save LaTeX table for the main radar-gauge occurrence contingency table.
    """
    tab = pd.crosstab(
        df_all["gauge_occurrence"],
        df_all["radar_occurrence"],
        normalize="all",
    )
    tab = tab.reindex(index=[0, 1], columns=[0, 1], fill_value=0.0)

    p00 = tab.loc[0, 0]
    p01 = tab.loc[0, 1]
    p10 = tab.loc[1, 0]
    p11 = tab.loc[1, 1]

    latex = rf"""
\begin{{table}}[H]
\centering
\caption{{Joint occurrence of rainfall in rain gauges and radar data. Values are proportions of all raw observations, before filtering positive rainfall intensities.}}
\label{{tab:radar-gauge-occurrence-raw}}
\begin{{threeparttable}}
\begin{{tabular}}{{lcc}}
\toprule
 & \multicolumn{{2}}{{c}}{{Radar occurrence}} \\
\cmidrule(lr){{2-3}}
Gauge occurrence & Radar dry & Radar wet \\
\midrule
Gauge dry & {p00:.3f} & {p01:.3f} \\
Gauge wet & {p10:.3f} & {p11:.3f} \\
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}
\small
\item The rain gauge is considered wet when \(Y_{{\mathrm{{obs}}}}>0\). 
The radar is considered wet when at least one associated COMEPHORE predictor is positive. 
The table is computed before filtering observations for the positive-intensity EGPD model.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""

    out_path = os.path.join(TAB_DIR, "occurrence_contingency_raw.tex")
    with open(out_path, "w") as f:
        f.write(latex)

    print("Saved LaTeX occurrence table:", out_path)
    print(latex)

    return latex

# %%
def fig_occurrence_contingency(df_all: pd.DataFrame):
    tab = pd.crosstab(
        df_all["gauge_occurrence"],
        df_all["radar_occurrence"],
        normalize="all",
    )
    tab = tab.reindex(index=[0, 1], columns=[0, 1], fill_value=0.0)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(tab.values)
    plt.colorbar(im, ax=ax, label="Proportion of observations")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Radar dry", "Radar wet"])
    ax.set_yticklabels(["Gauge dry", "Gauge wet"])
    ax.set_xlabel("Radar occurrence")
    ax.set_ylabel("Gauge occurrence")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{tab.values[i, j]:.3f}", ha="center", va="center")

    plt.tight_layout()
    savefig(fig, "01_occurrence_contingency")
    plt.close(fig)


def fig_occurrence_rates_by_month_hour(df_all: pd.DataFrame):
    monthly = (
        df_all.groupby("month")
        .agg(
            gauge_occurrence_rate=("gauge_occurrence", "mean"),
            radar_occurrence_rate=("radar_occurrence", "mean"),
            joint_occurrence_rate=("corres", "mean"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly["month"], monthly["gauge_occurrence_rate"], marker="o", label="Gauge wet")
    ax.plot(monthly["month"], monthly["radar_occurrence_rate"], marker="o", label="Radar wet")
    ax.plot(monthly["month"], monthly["joint_occurrence_rate"], marker="o", label="Gauge and radar wet")
    ax.set_xlabel("Month")
    ax.set_ylabel("Occurrence rate")
    ax.legend()
    add_grid(ax)
    plt.tight_layout()
    savefig(fig, "02_monthly_occurrence_rates")
    plt.close(fig)

    hourly = (
        df_all.groupby("hour")
        .agg(
            gauge_occurrence_rate=("gauge_occurrence", "mean"),
            radar_occurrence_rate=("radar_occurrence", "mean"),
            joint_occurrence_rate=("corres", "mean"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hourly["hour"], hourly["gauge_occurrence_rate"], marker="o", label="Gauge wet")
    ax.plot(hourly["hour"], hourly["radar_occurrence_rate"], marker="o", label="Radar wet")
    ax.plot(hourly["hour"], hourly["joint_occurrence_rate"], marker="o", label="Gauge and radar wet")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Occurrence rate")
    ax.legend()
    add_grid(ax)
    plt.tight_layout()
    savefig(fig, "03_hourly_occurrence_rates")
    plt.close(fig)


def fig_positive_rainfall_distribution(df_pos: pd.DataFrame):
    y = df_pos["Y_obs"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y, bins=80, density=True, alpha=0.75)
    ax.set_xlabel(r"Positive gauge rainfall $Y_{obs}$")
    ax.set_ylabel("Density")
    add_grid(ax)
    plt.tight_layout()
    savefig(fig, "04_positive_rainfall_histogram")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log1p(y), bins=80, density=True, alpha=0.75)
    ax.set_xlabel(r"$\log(1 + Y_{obs})$")
    ax.set_ylabel("Density")
    add_grid(ax)
    plt.tight_layout()
    savefig(fig, "05_positive_rainfall_log_histogram")
    plt.close(fig)


def fig_survival_positive_rainfall(df_pos: pd.DataFrame):
    y = df_pos["Y_obs"].to_numpy(float)
    y_sorted = np.sort(y)
    surv = 1.0 - np.arange(1, len(y_sorted) + 1) / len(y_sorted)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.step(y_sorted, surv, where="post")
    ax.set_yscale("log")
    ax.set_xlabel(r"Positive gauge rainfall $Y_{obs}$")
    ax.set_ylabel("Empirical survival probability")
    add_grid(ax)
    plt.tight_layout()
    savefig(fig, "06_positive_rainfall_survival")
    plt.close(fig)

    qs = [0.90, 0.95, 0.975, 0.99]
    qvals = np.quantile(y, qs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.step(y_sorted, surv, where="post")
    ax.set_yscale("log")

    for q, qv in zip(qs, qvals):
        ax.axvline(qv, linestyle="--", alpha=0.7)
        ax.text(qv, 0.05, f"q{q:.3f}={qv:.2f}", rotation=90, va="bottom", ha="right")

    ax.set_xlabel(r"Positive gauge rainfall $Y_{obs}$")
    ax.set_ylabel("Empirical survival probability")
    add_grid(ax)
    plt.tight_layout()
    savefig(fig, "07_positive_rainfall_survival_quantiles")
    plt.close(fig)


def fig_tipping_bucket_discretization(df_pos: pd.DataFrame):
    y = df_pos["Y_obs"].to_numpy(float)
    y_small = y[y <= 2.0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_small, bins=80, alpha=0.8)

    for k in range(1, 11):
        ax.axvline(k * BUCKET_RESOLUTION, linestyle="--", alpha=0.3)

    ax.set_xlabel(r"Positive gauge rainfall $Y_{obs}$ up to 2 mm")
    ax.set_ylabel("Count")
    add_grid(ax)
    plt.tight_layout()
    savefig(fig, "08_tipping_bucket_discretization")
    plt.close(fig)


def fig_top_correlations(df_pos: pd.DataFrame, x_cols_all: list[str]):
    corrs = (
        df_pos[["Y_obs"] + x_cols_all]
        .corr(numeric_only=True)["Y_obs"]
        .drop("Y_obs")
        .sort_values(ascending=False)
    )
    top_corrs = corrs.head(20)
    top_corrs_plot = top_corrs.copy()
    top_corrs_plot.index = [pretty_covariate_name(v) for v in top_corrs_plot.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    top_corrs_plot.sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel(r"Pearson correlation with $Y_{obs}$")
    ax.set_ylabel("")
    add_grid(ax, axis="x")
    plt.tight_layout()
    savefig(fig, "09_top_predictor_correlations")
    plt.close(fig)


def fig_lag_correlations(df_pos: pd.DataFrame, x_cols27: list[str]):
    lag_corrs = []

    for lag in ["-1h", "0h", "+1h"]:
        cols = [c for c in x_cols27 if c.endswith(f"dt{lag}")]
        vals = df_pos[["Y_obs"] + cols].corr(numeric_only=True)["Y_obs"].drop("Y_obs")
        for c, v in vals.items():
            lag_corrs.append({"lag": lag, "covariate": c, "corr": v})

    lag_corrs = pd.DataFrame(lag_corrs)
    lag_corrs.to_csv(os.path.join(TAB_DIR, "correlations_by_lag.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    lag_corrs.boxplot(column="corr", by="lag", ax=ax)
    ax.set_xlabel("Radar temporal lag")
    ax.set_ylabel(r"Pearson correlation with $Y_{obs}$")
    plt.tight_layout()
    savefig(fig, "10_correlation_by_radar_lag")
    plt.close(fig)


def fig_scatter_radar_gauge(df_pos: pd.DataFrame, predictor: str = "radar_max"):
    if predictor not in df_pos.columns:
        predictor = "radar_max"

    x = df_pos[predictor].to_numpy(float)
    y = df_pos["Y_obs"].to_numpy(float)
    label = pretty_covariate_name(predictor)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=6, alpha=0.2)
    ax.set_xlabel(label)
    ax.set_ylabel(r"Gauge rainfall $Y_{obs}$")
    add_grid(ax)
    plt.tight_layout()
    savefig(fig, f"11_scatter_y_vs_{predictor}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(np.log1p(x), np.log1p(y), s=6, alpha=0.2)
    ax.set_xlabel(f"log(1 + {label})")
    ax.set_ylabel(r"$\log(1 + Y_{obs})$")
    add_grid(ax)
    plt.tight_layout()
    savefig(fig, f"12_log_scatter_y_vs_{predictor}")
    plt.close(fig)


def fig_conditional_distribution_by_radar(df_pos: pd.DataFrame):
    df_plot = df_pos.copy()
    df_plot["radar_bin"] = pd.qcut(df_plot["radar_max"], q=8, duplicates="drop")

    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot.boxplot(column="Y_obs", by="radar_bin", ax=ax, showfliers=False)
    ax.set_xlabel("Radar maximum quantile bin")
    ax.set_ylabel(r"Gauge rainfall $Y_{obs}$")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    savefig(fig, "13_boxplot_y_by_radar_bin")
    plt.close(fig)

    df_plot["log_Y_obs"] = np.log1p(df_plot["Y_obs"])

    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot.boxplot(column="log_Y_obs", by="radar_bin", ax=ax, showfliers=False)
    ax.set_xlabel("Radar maximum quantile bin")
    ax.set_ylabel(r"$\log(1 + Y_{obs})$")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    savefig(fig, "14_log_boxplot_y_by_radar_bin")
    plt.close(fig)


def fig_conditional_quantiles_by_radar(df_pos: pd.DataFrame):
    df_plot = df_pos.copy()
    df_plot["radar_bin"] = pd.qcut(df_plot["radar_max"], q=10, duplicates="drop")

    cond = (
        df_plot.groupby("radar_bin", observed=True)
        .agg(
            radar_mid=("radar_max", "median"),
            y_med=("Y_obs", "median"),
            y_q75=("Y_obs", lambda x: np.quantile(x, 0.75)),
            y_q90=("Y_obs", lambda x: np.quantile(x, 0.90)),
            y_q95=("Y_obs", lambda x: np.quantile(x, 0.95)),
            n=("Y_obs", "size"),
        )
        .reset_index()
    )
    cond.to_csv(os.path.join(TAB_DIR, "conditional_quantiles_by_radar_max.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cond["radar_mid"], cond["y_med"], marker="o", label="median")
    ax.plot(cond["radar_mid"], cond["y_q75"], marker="o", label="q75")
    ax.plot(cond["radar_mid"], cond["y_q90"], marker="o", label="q90")
    ax.plot(cond["radar_mid"], cond["y_q95"], marker="o", label="q95")
    ax.set_xlabel("Median radar maximum in bin")
    ax.set_ylabel(r"Conditional quantiles of $Y_{obs}$")
    ax.legend()
    add_grid(ax)
    plt.tight_layout()
    savefig(fig, "15_conditional_quantiles_by_radar_max")
    plt.close(fig)


def fig_temporal_intensity_patterns(df_pos: pd.DataFrame):
    hourly = (
        df_pos.groupby("hour")["Y_obs"]
        .agg(["count", "mean", "median", lambda x: np.quantile(x, 0.95)])
    )
    hourly.columns = ["count", "mean", "median", "q95"]
    hourly.to_csv(os.path.join(TAB_DIR, "positive_intensity_by_hour.csv"))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hourly.index, hourly["mean"], marker="o", label="mean")
    ax.plot(hourly.index, hourly["q95"], marker="o", label="q95")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(r"Positive gauge rainfall $Y_{obs}$")
    ax.legend()
    add_grid(ax)
    plt.tight_layout()
    savefig(fig, "16_hourly_positive_intensity")
    plt.close(fig)

    monthly = (
        df_pos.groupby("month")["Y_obs"]
        .agg(["count", "mean", "median", lambda x: np.quantile(x, 0.95)])
    )
    monthly.columns = ["count", "mean", "median", "q95"]
    monthly.to_csv(os.path.join(TAB_DIR, "positive_intensity_by_month.csv"))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly.index, monthly["mean"], marker="o", label="mean")
    ax.plot(monthly.index, monthly["q95"], marker="o", label="q95")
    ax.set_xlabel("Month")
    ax.set_ylabel(r"Positive gauge rainfall $Y_{obs}$")
    ax.legend()
    add_grid(ax)
    plt.tight_layout()
    savefig(fig, "17_monthly_positive_intensity")
    plt.close(fig)


def fig_station_spatial_patterns(df_all: pd.DataFrame):
    df_plot = df_all.copy()
    df_plot["error_naive"] = df_plot["Y_obs"] - df_plot["radar_max"]

    station_stats = (
        df_plot.groupby("station")
        .agg(
            lat_Y=("lat_Y", "first"),
            lon_Y=("lon_Y", "first"),
            mean_error=("error_naive", "mean"),
            mae=("error_naive", lambda x: np.mean(np.abs(x))),
            gauge_occurrence_rate=("gauge_occurrence", "mean"),
            positive_mean=("Y_obs", lambda x: np.mean(x[x > 0]) if np.any(x > 0) else np.nan),
            n=("error_naive", "size"),
        )
        .reset_index()
    )
    station_stats.to_csv(os.path.join(TAB_DIR, "station_spatial_summaries.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        station_stats["lon_Y"],
        station_stats["lat_Y"],
        c=station_stats["mean_error"],
        s=80,
    )
    plt.colorbar(sc, ax=ax, label=r"Mean error $Y_{obs}$ - radar maximum")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    add_grid(ax)
    plt.tight_layout()
    savefig(fig, "18_station_mean_radar_gauge_error")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        station_stats["lon_Y"],
        station_stats["lat_Y"],
        c=station_stats["gauge_occurrence_rate"],
        s=80,
    )
    plt.colorbar(sc, ax=ax, label="Gauge occurrence rate")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    add_grid(ax)
    plt.tight_layout()
    savefig(fig, "19_station_gauge_occurrence_rate")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        station_stats["lon_Y"],
        station_stats["lat_Y"],
        c=station_stats["positive_mean"],
        s=80,
    )
    plt.colorbar(sc, ax=ax, label=r"Mean positive $Y_{obs}$")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    add_grid(ax)
    plt.tight_layout()
    savefig(fig, "20_station_positive_mean_intensity")
    plt.close(fig)


def fig_example_events(df_all: pd.DataFrame, n_events: int = 3):
    """
    Select a few intense days and plot gauge rainfall and radar summaries.
    This is useful to visually illustrate timing and intensity discrepancies.
    """
    d = df_all.copy()
    d["date"] = d["time"].dt.floor("D")

    daily = (
        d.groupby("date")
        .agg(
            y_sum=("Y_obs", "sum"),
            y_max=("Y_obs", "max"),
            radar_max=("radar_max", "max"),
        )
        .sort_values("y_sum", ascending=False)
        .head(n_events)
        .reset_index()
    )
    daily.to_csv(os.path.join(TAB_DIR, "selected_example_event_days.csv"), index=False)

    for i, row in daily.iterrows():
        day = row["date"]
        mask = d["date"] == day
        sub = d.loc[mask].sort_values("time")

        # Average over stations at each time for a compact event-level display.
        ts = (
            sub.groupby("time")
            .agg(
                gauge_mean=("Y_obs", "mean"),
                gauge_max=("Y_obs", "max"),
                radar_max=("radar_max", "mean"),
                radar_mean=("radar_mean", "mean"),
            )
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ts["time"], ts["gauge_mean"], label="Gauge mean")
        ax.plot(ts["time"], ts["gauge_max"], label="Gauge max")
        ax.plot(ts["time"], ts["radar_max"], label="Radar maximum, mean over stations")
        ax.plot(ts["time"], ts["radar_mean"], label="Radar mean, mean over stations")
        ax.set_xlabel("Time")
        ax.set_ylabel("Rainfall")
        ax.legend()
        add_grid(ax)
        plt.tight_layout()
        savefig(fig, f"21_example_event_{i+1}_{pd.Timestamp(day).date()}")
        plt.close(fig)


# %%
df_raw = pd.read_csv(DOWNSCALING_TABLE, sep=";")
df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True)

print("Raw shape:", df_raw.shape)
print("Missing values in Y_obs:", df_raw["Y_obs"].isna().sum())
print("Proportion of Y_obs = 0:", (df_raw["Y_obs"] == 0).mean())

df_all, df_pos, x_cols27, x_cols_all = prepare_analysis_dataframe(df_raw)

print("Prepared df_all shape:", df_all.shape)
print("Prepared df_pos shape:", df_pos.shape)
print("Number of radar pixel-time covariates:", len(x_cols27))
print("Number of all candidate covariates:", len(x_cols_all))

df_all.to_csv(os.path.join(TAB_DIR, "prepared_all_nonmissing_y.csv"), index=False)
df_pos.to_csv(os.path.join(TAB_DIR, "prepared_positive_with_radar.csv"), index=False)

save_summary_tables(df_all, df_pos, x_cols_all)

summary, corrs_df, monthly_occ, hourly_occ, station_occ = save_summary_tables(
    df_all, df_pos, x_cols_all
)

occ_def_comparison = save_occurrence_definition_comparison(df_all)
save_occurrence_latex_table(df_all)

print("\n=== Radar occurrence definition comparison ===")
print(occ_def_comparison.to_string(index=False))

fig_occurrence_contingency(df_all)
fig_occurrence_rates_by_month_hour(df_all)

# Figures on positive rainfall intensities.
fig_positive_rainfall_distribution(df_pos)
fig_survival_positive_rainfall(df_pos)
fig_tipping_bucket_discretization(df_pos)

# Radar-gauge relationship.
fig_top_correlations(df_pos, x_cols_all)
fig_lag_correlations(df_pos, x_cols27)
fig_scatter_radar_gauge(df_pos, predictor="radar_max")

# Also plot the most correlated radar covariate if available.
corrs = (
    df_pos[["Y_obs"] + x_cols_all]
    .corr(numeric_only=True)["Y_obs"]
    .drop("Y_obs")
    .sort_values(ascending=False)
)
best_cov = corrs.index[0]
fig_scatter_radar_gauge(df_pos, predictor=best_cov)

# Conditional distributions.
fig_conditional_distribution_by_radar(df_pos)
fig_conditional_quantiles_by_radar(df_pos)

# Temporal and spatial summaries.
fig_temporal_intensity_patterns(df_pos)
fig_station_spatial_patterns(df_all)

# Example event days.
fig_example_events(df_all, n_events=3)
# %%
