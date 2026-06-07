# %%
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from downscaling.paths import DOWNSCALING_TABLE, IM_FOLDER
from downscaling.plotting import configure_plot_style, save_png
from downscaling.features import standardize_train_only
from downscaling.occurrence import (
    prepare_occurrence_dataframe,
    build_Xy_occurrence,
    train_logit_model,
    predict_occurrence_probability,
    plot_loss_history,
)
from downscaling.occurrence_metrics import (
    evaluate_occurrence_predictions,
    plot_validation_summary,
    plot_roc_curve,
)

configure_plot_style()

OUT_DIR = Path(IM_FOLDER) / "leave_one_site_out_occurrence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2026
BLOCK = "15D"

STATION_COL_CANDIDATES = [
    "site", "station", "station_name", "gauge",
    "name_Y", "site_Y", "id_Y"
]


def savefig(fig, name):
    save_png(fig, OUT_DIR / name)


def find_station_col(df):
    for col in STATION_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        "No station column found. Please set STATION_COL manually.\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


def calibration_table(y, p, n_bins=10):
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    return (
        df.groupby("bin", observed=True)
        .agg(
            n=("y", "size"),
            mean_pred=("p", "mean"),
            obs_freq=("y", "mean"),
        )
        .reset_index()
    )


def summarize_metrics(name, y, p, p_base, n_params, station):
    m = evaluate_occurrence_predictions(
        y_true=y,
        p_pred=p,
        p_null=p_base,
        n_params=n_params,
        threshold=0.5,
        calib_bins=10,
    )

    mb = evaluate_occurrence_predictions(
        y_true=y,
        p_pred=p_base,
        p_null=p_base,
        n_params=1,
        threshold=0.5,
        calib_bins=10,
    )

    return {
        "left_out_station": station,
        "split": name,
        "n": m["n"],
        "rain_freq": m["rain_freq"],
        "mean_pred": m["mean_pred"],
        "brier_model": m["brier"],
        "brier_baseline": mb["brier"],
        "brier_gain": mb["brier"] - m["brier"],
        "logloss_model": m["logloss"],
        "logloss_baseline": mb["logloss"],
        "logloss_gain": mb["logloss"] - m["logloss"],
        "auc": m["auc"],
        "ece": m["ece_quantile"],
    }


def plot_prob_distribution(y, p, p0, station, filename):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(p[y == 0], bins=50, alpha=0.6, density=True, label="No rain")
    ax.hist(p[y == 1], bins=50, alpha=0.6, density=True, label="Rain")
    ax.axvline(p0, linestyle="--", label="Train rain frequency")
    ax.set_xlabel(r"Predicted $P(Y_{\mathbf{s},t}>0 \mid \mathbf{C}_{\mathbf{s},t})$")
    ax.set_ylabel("Density")
    ax.set_title(f"Left-out station: {station}")
    ax.legend()
    fig.tight_layout()
    save_png(fig, filename)
    plt.close(fig)


def plot_reliability_curve(y, p, station, filename, n_bins=10):
    tab = calibration_table(y, p, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(tab["mean_pred"], tab["obs_freq"], marker="o", label="LOSO")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed rain frequency")
    ax.set_title(f"Reliability curve — {station}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_png(fig, filename)
    plt.close(fig)


# %%
df_raw = pd.read_csv(DOWNSCALING_TABLE, sep=";")
df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True)

df_occ, x_cols27, x_cols = prepare_occurrence_dataframe(
    df_raw,
    use_time=True,
    use_spatial=True,
    use_summaries=True,
    use_cube=True,
)

df_occ["is_rain"] = (df_occ["Y_obs"] > 0).astype(int)

STATION_COL = find_station_col(df_occ)
print("Using station column:", STATION_COL)

stations = sorted(df_occ[STATION_COL].dropna().unique())
print("Number of stations:", len(stations))
print(stations)

# %%
all_metrics = []
all_preds = []

for i, station in enumerate(stations, start=1):
    print(f"\n[{i}/{len(stations)}] Leaving out station: {station}")

    train_idx = df_occ.index[df_occ[STATION_COL] != station]
    test_idx = df_occ.index[df_occ[STATION_COL] == station]

    if len(test_idx) < 20:
        print(f"Skipping {station}: too few observations.")
        continue

    rain_freq_test = df_occ.loc[test_idx, "is_rain"].mean()
    if rain_freq_test == 0 or rain_freq_test == 1:
        print(f"Warning: station {station} has degenerate occurrence frequency.")

    df_std, mu, sdv = standardize_train_only(
        df_occ,
        train_idx,
        x_cols,
    )

    built = build_Xy_occurrence(
        df_std=df_std,
        x_cols=x_cols,
        train_idx_labels=train_idx,
        valid_idx_labels=test_idx,
    )

    fit = train_logit_model(
        X_train=built["X_train"],
        y_train=built["y_train"],
        X_valid=built["X_valid"],
        y_valid=built["y_valid"],
        lr=1e-3,
        n_epochs=300,
        seed=SEED + i,
        device=None,
        patience=60,
    )

    y_test = built["y_valid"].reshape(-1)
    p_test = predict_occurrence_probability(
        fit["model"],
        built["X_valid"],
    )

    p0_train = float(built["y_train"].mean())
    p_test_base = np.full_like(y_test, p0_train, dtype=float)

    n_params = 1 + built["X_train"].shape[1]

    row = summarize_metrics(
        name="loso_test",
        y=y_test,
        p=p_test,
        p_base=p_test_base,
        n_params=n_params,
        station=station,
    )

    row["train_rain_freq"] = p0_train
    row["stopped_epoch"] = fit.get("stopped_epoch", np.nan)
    all_metrics.append(row)

    df_pred_station = df_occ.loc[test_idx].copy()
    df_pred_station["left_out_station"] = station
    df_pred_station["split"] = "loso_test"
    df_pred_station["y_occ"] = y_test
    df_pred_station["p_occ_hat"] = p_test
    df_pred_station["p_occ_baseline"] = p0_train

    all_preds.append(df_pred_station)

    safe_station = str(station).replace(" ", "_").replace("/", "_")

    plot_prob_distribution(
        y=y_test,
        p=p_test,
        p0=p0_train,
        station=station,
        filename=OUT_DIR / f"loso_occurrence_probability_distribution_{safe_station}.png",
    )

    plot_reliability_curve(
        y=y_test,
        p=p_test,
        station=station,
        filename=OUT_DIR / f"loso_occurrence_reliability_{safe_station}.png",
    )

    plot_validation_summary(
        y_true=y_test,
        p_pred=p_test,
        title_prefix=f"LOSO {station}",
        filename=OUT_DIR / f"loso_occurrence_summary_{safe_station}.png",
    )

    plot_roc_curve(
        y_true=y_test,
        p_pred=p_test,
        filename=OUT_DIR / f"loso_occurrence_roc_{safe_station}.png",
    )


# %%
df_metrics_loso = pd.DataFrame(all_metrics)
df_preds_loso = pd.concat(all_preds, ignore_index=True)

df_metrics_loso.to_csv(
    OUT_DIR / "loso_occurrence_metrics_by_station.csv",
    index=False,
)

df_preds_loso.to_csv(
    OUT_DIR / "loso_occurrence_predictions.csv",
    index=False,
)

print("\nLOSO occurrence metrics by station:")
print(df_metrics_loso.to_string(index=False))

# %%
summary_loso = pd.DataFrame({
    "n_sites": [df_metrics_loso["left_out_station"].nunique()],
    "brier_model_mean": [df_metrics_loso["brier_model"].mean()],
    "brier_model_sd": [df_metrics_loso["brier_model"].std()],
    "brier_gain_mean": [df_metrics_loso["brier_gain"].mean()],
    "brier_gain_sd": [df_metrics_loso["brier_gain"].std()],
    "logloss_model_mean": [df_metrics_loso["logloss_model"].mean()],
    "logloss_gain_mean": [df_metrics_loso["logloss_gain"].mean()],
    "auc_mean": [df_metrics_loso["auc"].mean()],
    "auc_sd": [df_metrics_loso["auc"].std()],
    "ece_mean": [df_metrics_loso["ece"].mean()],
    "ece_sd": [df_metrics_loso["ece"].std()],
    "rain_freq_mean": [df_metrics_loso["rain_freq"].mean()],
    "mean_pred_mean": [df_metrics_loso["mean_pred"].mean()],
})

summary_loso.to_csv(
    OUT_DIR / "loso_occurrence_summary.csv",
    index=False,
)

print("\nLOSO occurrence summary:")
print(summary_loso.to_string(index=False))

# %%
# Global pooled diagnostics

y_all = df_preds_loso["y_occ"].to_numpy(int)
p_all = df_preds_loso["p_occ_hat"].to_numpy(float)
p0_all = float(df_preds_loso["p_occ_baseline"].mean())

plot_prob_distribution(
    y=y_all,
    p=p_all,
    p0=p0_all,
    station="all stations pooled",
    filename=OUT_DIR / "loso_occurrence_probability_distribution_pooled.png",
)

plot_reliability_curve(
    y=y_all,
    p=p_all,
    station="all stations pooled",
    filename=OUT_DIR / "loso_occurrence_reliability_pooled.png",
)

plot_validation_summary(
    y_true=y_all,
    p_pred=p_all,
    title_prefix="LOSO pooled",
    filename=OUT_DIR / "loso_occurrence_summary_pooled.png",
)

plot_roc_curve(
    y_true=y_all,
    p_pred=p_all,
    filename=OUT_DIR / "loso_occurrence_roc_pooled.png",
)

# %%
# Boxplots over left-out sites

for col in ["brier_model", "brier_gain", "logloss_model", "logloss_gain", "auc", "ece"]:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.boxplot(df_metrics_loso[col].dropna(), showmeans=True)
    ax.set_ylabel(col)
    ax.set_xticks([1])
    ax.set_xticklabels(["LOSO sites"])
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, f"loso_occurrence_boxplot_{col}.png")
    plt.show()

# %%
# Observed vs predicted rain frequency by station

plot_df = df_metrics_loso.sort_values("rain_freq").copy()
x = np.arange(len(plot_df))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, plot_df["rain_freq"], marker="o", label="Observed rain frequency")
ax.plot(x, plot_df["mean_pred"], marker="o", label="Mean predicted probability")
ax.set_xticks(x)
ax.set_xticklabels(plot_df["left_out_station"], rotation=45, ha="right")
ax.set_ylabel("Probability")
ax.set_title("LOSO occurrence: observed vs predicted rain frequency")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
savefig(fig, "loso_occurrence_observed_vs_predicted_frequency_by_station.png")
plt.show()

print("\nDone. Outputs are in:", OUT_DIR)