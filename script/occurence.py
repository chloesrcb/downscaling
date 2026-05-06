#%%
import os
import numpy as np
import pandas as pd
import torch

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from downscaling.paths import DOWNSCALING_TABLE, IM_FOLDER
from downscaling.splits import make_single_split_from_train, make_blocked_cv_splits
from downscaling.features import standardize_train_only

from downscaling.occurrence import (
    prepare_occurrence_dataframe,
    build_Xy_occurrence,
    train_logit_model,
    predict_occurrence_probability,
)

from downscaling.occurrence_metrics import (
    evaluate_occurrence_predictions,
    plot_validation_summary,
    plot_loss_history,
)


from downscaling.splits import make_split_blocked
from downscaling.data import get_x_cols27_downscaling

#%%

# Load raw data
df_raw = pd.read_csv(DOWNSCALING_TABLE, sep=";")
df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True)

print("Missing values in Y_obs:", df_raw["Y_obs"].isna().sum())
print("Proportion of Y_obs = 0:", (df_raw["Y_obs"] == 0).mean())

#%%
# Build the binary occurrence dataset
df_occ, x_cols27, x_cols = prepare_occurrence_dataframe(
    df_raw,
    use_time=True,
    use_spatial=True,
    use_summaries=True,
    use_cube=True,
)

print("Occurrence dataset shape:", df_occ.shape)
print("Number of predictors:", len(x_cols))

#%%
# Create a blocked train/validation split
split = make_split_blocked(df_occ, train_frac=0.8, seed=2, block="30D")
print("n_train:", len(split["train_idx"]), "n_valid:", len(split["valid_idx"]))

#%%
# Standardize predictors using train only
df_std, mu, sdv = standardize_train_only(df_occ, split["train_idx"], x_cols)

built = build_Xy_occurrence(
    df_std=df_std,
    x_cols=x_cols,
    train_idx_labels=split["train_idx"],
    valid_idx_labels=split["valid_idx"],
)

print("X_train shape:", built["X_train"].shape)
print("Rain frequency train:", built["y_train"].mean())
print("Rain frequency valid:", built["y_valid"].mean())

#%%
# Fit the logistic occurrence model
fit = train_logit_model(
    X_train=built["X_train"],
    y_train=built["y_train"],
    X_valid=built["X_valid"],
    y_valid=built["y_valid"],
    lr=1e-2,
    n_epochs=400,
    seed=1,
    device=None,
    patience=40,
)

print("Stopped epoch:", fit["stopped_epoch"])

#%%
# Inspect fitted coefficients and odds ratios
with torch.no_grad():
    bias_hat = fit["model"].linear.bias.detach().cpu().item()
    w = fit["model"].linear.weight.detach().cpu().numpy().ravel()

coef_df = pd.DataFrame({
    "feature": x_cols,
    "coef_logit": w,
    "odds_ratio_1sd": np.exp(w),
    "abs_coef": np.abs(w)
}).sort_values("abs_coef", ascending=False)

print("Estimated intercept:", bias_hat)
print("Intercept probability:", 1 / (1 + np.exp(-bias_hat)))
print(coef_df.head(20))

#%%
plot_loss_history(
    fit,
    title="Occurrence logistic model loss",
    filename=os.path.join(IM_FOLDER, "occurrence_logit_loss.png")
)

#%%
# Predict on validation
y_valid = built["y_valid"].reshape(-1)
p_valid = predict_occurrence_probability(fit["model"], built["X_valid"])

# Build the intercept-only baseline on the same validation set
p0 = float(built["y_train"].mean())
p_valid_base = np.full_like(y_valid, fill_value=p0, dtype=float)

# Number of free parameters in the fitted logistic model
# one intercept + one coefficient per predictor
n_params_model = 1 + built["X_train"].shape[1]

metrics_valid = evaluate_occurrence_predictions(
    y_true=y_valid,
    p_pred=p_valid,
    p_null=p_valid_base,
    n_params=n_params_model,
    threshold=0.5,
    calib_bins=10
)

metrics_base = evaluate_occurrence_predictions(
    y_true=y_valid,
    p_pred=p_valid_base,
    p_null=p_valid_base,
    n_params=1,
    threshold=0.5,
    calib_bins=10
)

print("Validation metrics for the fitted model")
for k, v in metrics_valid.items():
    print(f"{k}: {v}")

print("\nValidation metrics for the constant baseline")
for k, v in metrics_base.items():
    print(f"{k}: {v}")

#%%
# Better validation graph
rel_valid, roc_valid = plot_validation_summary(
    y_true=y_valid,
    p_pred=p_valid,
    title_prefix="Validation",
    filename=os.path.join(IM_FOLDER, "occurrence_validation_summary.png")
)

#%%
# Add probabilities back to the full dataset
p_all = predict_occurrence_probability(fit["model"], built["X_all"])

df_occ_pred = df_occ.copy()
df_occ_pred["p_occ_hat"] = p_all
df_occ_pred["is_rain"] = (df_occ_pred["Y_obs"] > 0).astype(int)

print(df_occ_pred[["time", "station", "Y_obs", "is_rain", "p_occ_hat"]].head())

#%%
# Blocked cross-validation
K = 5
cv_splits = make_blocked_cv_splits(df_occ, n_splits=K, block="30D", seed=1)

rows_cv = []

for sp in cv_splits:
    df_std_cv, mu_cv, sdv_cv = standardize_train_only(df_occ, sp["train_idx"], x_cols)

    built_cv = build_Xy_occurrence(
        df_std=df_std_cv,
        x_cols=x_cols,
        train_idx_labels=sp["train_idx"],
        valid_idx_labels=sp["valid_idx"],
    )

    fit_cv = train_logit_model(
        X_train=built_cv["X_train"],
        y_train=built_cv["y_train"],
        X_valid=built_cv["X_valid"],
        y_valid=built_cv["y_valid"],
        lr=1e-2,
        n_epochs=300,
        seed=1,
        device=None,
        patience=30,
    )

    y_valid_cv = built_cv["y_valid"].reshape(-1)
    p_valid_cv = predict_occurrence_probability(fit_cv["model"], built_cv["X_valid"])

    p0_cv = float(built_cv["y_train"].mean())
    p_valid_base_cv = np.full_like(y_valid_cv, fill_value=p0_cv, dtype=float)

    n_params_cv = 1 + built_cv["X_train"].shape[1]

    metrics_cv = evaluate_occurrence_predictions(
        y_true=y_valid_cv,
        p_pred=p_valid_cv,
        p_null=p_valid_base_cv,
        n_params=n_params_cv,
        threshold=0.5,
        calib_bins=10
    )

    metrics_base_cv = evaluate_occurrence_predictions(
        y_true=y_valid_cv,
        p_pred=p_valid_base_cv,
        p_null=p_valid_base_cv,
        n_params=1,
        threshold=0.5,
        calib_bins=10
    )

    rows_cv.append({
        "fold": sp["fold"],
        "n_valid": metrics_cv["n"],
        "rain_freq_valid": metrics_cv["rain_freq"],
        "mean_pred_valid": metrics_cv["mean_pred"],
        "brier_model": metrics_cv["brier"],
        "brier_baseline": metrics_base_cv["brier"],
        "logloss_model": metrics_cv["logloss"],
        "logloss_baseline": metrics_base_cv["logloss"],
        "auc_model": metrics_cv["auc"],
        "auc_baseline": metrics_base_cv["auc"],
        "acc_model": metrics_cv["accuracy_050"],
        "acc_baseline": metrics_base_cv["accuracy_050"],
        "ece_model": metrics_cv["ece_quantile"],
        "ece_baseline": metrics_base_cv["ece_quantile"],
        "mcfadden_r2": metrics_cv["mcfadden_r2"],
        "lr_chi2": metrics_cv["lr_chi2"],
        "aic_model": metrics_cv["aic"],
        "bic_model": metrics_cv["bic"],
        "stopped_epoch": fit_cv["stopped_epoch"],
    })

df_cv = pd.DataFrame(rows_cv)
print(df_cv)

#%%
# Summarize CV performance
summary_cv = pd.DataFrame({
    "metric": [
        "brier_model",
        "brier_baseline",
        "logloss_model",
        "logloss_baseline",
        "auc_model",
        "acc_model",
        "ece_model",
        "mcfadden_r2",
        "lr_chi2",
        "aic_model",
        "bic_model",
    ],
    "mean": [
        df_cv["brier_model"].mean(),
        df_cv["brier_baseline"].mean(),
        df_cv["logloss_model"].mean(),
        df_cv["logloss_baseline"].mean(),
        df_cv["auc_model"].mean(),
        df_cv["acc_model"].mean(),
        df_cv["ece_model"].mean(),
        df_cv["mcfadden_r2"].mean(),
        df_cv["lr_chi2"].mean(),
        df_cv["aic_model"].mean(),
        df_cv["bic_model"].mean(),
    ],
    "std": [
        df_cv["brier_model"].std(ddof=1),
        df_cv["brier_baseline"].std(ddof=1),
        df_cv["logloss_model"].std(ddof=1),
        df_cv["logloss_baseline"].std(ddof=1),
        df_cv["auc_model"].std(ddof=1),
        df_cv["acc_model"].std(ddof=1),
        df_cv["ece_model"].std(ddof=1),
        df_cv["mcfadden_r2"].std(ddof=1),
        df_cv["lr_chi2"].std(ddof=1),
        df_cv["aic_model"].std(ddof=1),
        df_cv["bic_model"].std(ddof=1),
    ],
})
print(summary_cv)

#%%
# Check how many incoherent rows were removed
df_tmp = df_raw.loc[df_raw["Y_obs"].notna()].copy()

x_cols27_tmp = get_x_cols27_downscaling(df_tmp)
X_block_tmp = df_tmp[x_cols27_tmp].to_numpy(dtype=float)
x_cube_sum_tmp = X_block_tmp.sum(axis=1)

mask_incoherent = (df_tmp["Y_obs"] > 0) & (x_cube_sum_tmp == 0)

n_total = len(df_tmp)
n_removed = int(mask_incoherent.sum())
prop_removed = n_removed / n_total

print("Total observations:", n_total)
print("Removed observations:", n_removed)
print("Proportion removed:", prop_removed)

mask_positive = df_tmp["Y_obs"] > 0
n_positive = int(mask_positive.sum())
n_removed_among_positive = int(mask_incoherent.sum())
prop_removed_among_positive = n_removed_among_positive / n_positive if n_positive > 0 else np.nan

print("Positive-rain observations:", n_positive)
print("Removed among positive-rain cases:", n_removed_among_positive)
print("Proportion removed among positive-rain cases:", prop_removed_among_positive)

removed_by_station = (
    df_tmp.loc[mask_incoherent]
    .groupby("station")
    .size()
    .sort_values(ascending=False)
)

print(removed_by_station)