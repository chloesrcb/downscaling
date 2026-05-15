#%%
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from downscaling.paths import DOWNSCALING_TABLE, IM_FOLDER
from downscaling.features import standardize_train_only
from downscaling.splits import make_split_blocked
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
)

#%%
def plot_prob_distribution(y, p, p0, title, filename):
    plt.figure(figsize=(7, 4))
    plt.hist(p[y == 0], bins=50, alpha=0.6, density=True, label="No rain")
    plt.hist(p[y == 1], bins=50, alpha=0.6, density=True, label="Rain")
    plt.axvline(p0, linestyle="--", label="Train rain frequency")
    plt.xlabel("Predicted occurrence probability")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


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


def summarize_metrics(name, y, p, p_base, n_params):
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

#%%
# Load data

df_raw = pd.read_csv(DOWNSCALING_TABLE, sep=";")
df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True)

print("Raw shape:", df_raw.shape)
print("Time range:", df_raw["time"].min(), "->", df_raw["time"].max())
print("Missing Y_obs:", df_raw["Y_obs"].isna().sum())
print("Rain frequency raw:", (df_raw["Y_obs"] > 0).mean())

#%%
# Prepare occurrence dataframe

df_occ, x_cols27, x_cols = prepare_occurrence_dataframe(
    df_raw,
    use_time=True,
    use_spatial=True,
    use_summaries=True,
    use_cube=True,
)

df_occ["is_rain"] = (df_occ["Y_obs"] > 0).astype(int)

print("Occurrence shape:", df_occ.shape)
print("Number of predictors:", len(x_cols))
print("Rain frequency:", df_occ["is_rain"].mean())

#%%
# Outer split: train_valid / test
outer_split = make_split_blocked(
    df_occ,
    train_frac=0.90,
    seed=2026,
    block="7D",
)

train_valid_idx = outer_split["train_idx"]
test_idx = outer_split["valid_idx"]

print("\nOuter split")
print("n_train_valid:", len(train_valid_idx))
print("n_test:", len(test_idx))
print("rain freq train_valid:", df_occ.loc[train_valid_idx, "is_rain"].mean())
print("rain freq test:", df_occ.loc[test_idx, "is_rain"].mean())

#%%
# Inner split: train / valid

df_train_valid = df_occ.loc[train_valid_idx].copy()

inner_split = make_split_blocked(
    df_train_valid,
    train_frac=0.80,
    seed=2,
    block="7D",
)

train_idx = inner_split["train_idx"]
valid_idx = inner_split["valid_idx"]

print("\nInner split")
print("n_train:", len(train_idx))
print("n_valid:", len(valid_idx))
print("rain freq train:", df_occ.loc[train_idx, "is_rain"].mean())
print("rain freq valid:", df_occ.loc[valid_idx, "is_rain"].mean())

#%%
# Standardization
df_std, mu, sdv = standardize_train_only(
    df_occ,
    train_idx,
    x_cols,
)

built = build_Xy_occurrence(
    df_std=df_std,
    x_cols=x_cols,
    train_idx_labels=train_idx,
    valid_idx_labels=valid_idx,
)

#%%
# Fit model
fit = train_logit_model(
    X_train=built["X_train"],
    y_train=built["y_train"],
    X_valid=built["X_valid"],
    y_valid=built["y_valid"],
    lr=1e-3,
    n_epochs=300,
    seed=1,
    device=None,
    patience=60,
)

print("Stopped epoch:", fit["stopped_epoch"])

#%%
plot_loss_history(
    fit,
    title="",
    filename=os.path.join(IM_FOLDER, "occurrence_logit_loss.png"),
)

#%%
# Predictions on valid and test

y_valid = built["y_valid"].reshape(-1)
p_valid = predict_occurrence_probability(fit["model"], built["X_valid"])

built_test = build_Xy_occurrence(
    df_std=df_std,
    x_cols=x_cols,
    train_idx_labels=train_idx,
    valid_idx_labels=test_idx,
)

y_test = built_test["y_valid"].reshape(-1)
p_test = predict_occurrence_probability(fit["model"], built_test["X_valid"])

p0_train = float(built["y_train"].mean())
p_valid_base = np.full_like(y_valid, p0_train, dtype=float)
p_test_base = np.full_like(y_test, p0_train, dtype=float)

n_params = 1 + built["X_train"].shape[1]

#%%
# Metrics

rows = [
    summarize_metrics("valid", y_valid, p_valid, p_valid_base, n_params),
    summarize_metrics("test", y_test, p_test, p_test_base, n_params),
]

df_metrics = pd.DataFrame(rows)
print(df_metrics)

df_metrics.to_csv(
    os.path.join(IM_FOLDER, "occurrence_valid_test_metrics.csv"),
    index=False,
)

#%%
# Calibration table on test
cal_test = calibration_table(y_test, p_test, n_bins=10)
print(cal_test)

cal_test.to_csv(
    os.path.join(IM_FOLDER, "occurrence_test_calibration_table.csv"),
    index=False,
)

#%%
# Plots on subsample only
rng = np.random.default_rng(123)
n_plot = min(200_000, len(y_test))
idx_plot = rng.choice(len(y_test), size=n_plot, replace=False)

plot_validation_summary(
    y_true=y_test[idx_plot],
    p_pred=p_test[idx_plot],
    title_prefix="Test",
    filename=os.path.join(IM_FOLDER, "occurrence_test_summary.png"),
)

plot_prob_distribution(
    y=y_test[idx_plot],
    p=p_test[idx_plot],
    p0=p0_train,
    title="Test predicted probabilities",
    filename=os.path.join(IM_FOLDER, "occurrence_test_probability_distribution.png"),
)

#%%
from downscaling.occurrence_metrics import plot_roc_curve
plot_roc_curve(
    y_true=y_test[idx_plot],
    p_pred=p_test[idx_plot],
    filename=os.path.join(IM_FOLDER, "occurrence_test_roc_curve.png"),
)

#%%
# Save predictions

df_occ_pred = df_occ.copy()
df_occ_pred["p_occ_hat"] = np.nan
df_occ_pred.loc[valid_idx, "p_occ_hat"] = p_valid
df_occ_pred.loc[test_idx, "p_occ_hat"] = p_test

df_occ_pred["split"] = "train"
df_occ_pred.loc[valid_idx, "split"] = "valid"
df_occ_pred.loc[test_idx, "split"] = "test"

df_occ_pred.to_csv(
    os.path.join(IM_FOLDER, "occurrence_predictions_valid_test.csv"),
    index=False,
)