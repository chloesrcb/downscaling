# %%
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downscaling.config import (
    CENSOR_THRESHOLD,
    KAPPA_INIT,
    SIGMA_INIT,
    SINGLE_COV_COL,
    XI_INIT,
)
from downscaling.data import (
    make_covariate_sets,
    prepare_modeling_dataframe,
)
from downscaling.diagnostics import (
    add_prediction_quantities,
    plot_exponential_qq,
    plot_pit_histograms,
    plot_quantile_calibration,
    plot_tail_exceedance_calibration,
    summarize_model_comparison,
)
from downscaling.evaluation import (
    evaluate_fixed_nn_model,
    evaluate_single_covariate_model,
    evaluate_stationary_candidate,
)
from downscaling.paths import (
    DOWNSCALING_TABLE,
    IM_FOLDER,
    OUT_BEST_PARAMS,
    OUT_COMPARISON,
    OUT_RERANK,
    OUT_SUMMARY,
    OUT_SUMMARY_DELTA,
    OUT_TEST,
    OUT_TEST_PRED,
    OUT_TUNING,
    make_output_dirs,
)
from downscaling.prediction import (
    fit_predict_nn_test,
    fit_predict_regression_test,
    fit_predict_stationary_test,
)
from downscaling.splits import (
    make_blocked_cv_splits,
    make_train_valid_test_split,
)
from downscaling.tuning import (
    rerank_top_nn_configs,
    tune_nn_on_outer_train,
)

make_output_dirs()
IM_FOLDER = Path(IM_FOLDER)


# %%
# EGPD probabilities, quantiles, simulation, and tail scores
def egpd_cdf(y, sigma, kappa, xi, eps=1e-12):
    y = np.maximum(np.asarray(y), 0.0)
    sigma = np.maximum(np.asarray(sigma), eps)
    kappa = np.maximum(np.asarray(kappa), eps)
    xi = np.asarray(xi)

    base = np.where(
        np.abs(xi) > eps,
        1.0 - (1.0 + xi * y / sigma) ** (-1.0 / xi),
        1.0 - np.exp(-y / sigma),
    )

    base = np.clip(base, eps, 1.0 - eps)

    return base ** kappa


def egpd_quantile(p, sigma, kappa, xi, eps=1e-12):
    p = np.clip(np.asarray(p), eps, 1.0 - eps)
    sigma = np.maximum(np.asarray(sigma), eps)
    kappa = np.maximum(np.asarray(kappa), eps)
    xi = np.asarray(xi)

    base_p = p ** (1.0 / kappa)

    return np.where(
        np.abs(xi) > eps,
        sigma / xi * ((1.0 - base_p) ** (-xi) - 1.0),
        -sigma * np.log(1.0 - base_p),
    )


def tail_weighted_threshold_score(pred_df, alpha=1.0, q_low=0.95, q_high=0.995, n_thresholds=24):
    y = pred_df["Y_obs"].to_numpy()

    thresholds = np.quantile(y, np.linspace(q_low, q_high, n_thresholds))
    thresholds = np.unique(thresholds)

    if len(thresholds) == 0:
        return np.nan

    u0 = thresholds[0]
    if u0 <= 0:
        u0 = np.min(thresholds[thresholds > 0])

    weights = (thresholds / u0) ** alpha
    weights = weights / weights.sum()

    sigma = pred_df["sigma"].to_numpy()
    kappa = pred_df["kappa"].to_numpy()
    xi = pred_df["xi"].to_numpy()

    score = 0.0

    for w, u in zip(weights, thresholds):
        obs_exceed = (y > u).astype(float)
        pred_exceed = 1.0 - egpd_cdf(u, sigma, kappa, xi)
        score += w * np.mean((obs_exceed - pred_exceed) ** 2)

    return score


def add_twcrps_alpha_scores(pred_df, model_col="model", alphas=(0.0, 1.0, 2.0)):
    rows = []

    for model, d in pred_df.groupby(model_col):
        row = {"model": model}

        for alpha in alphas:
            row[f"twcrps_alpha_{alpha:g}"] = tail_weighted_threshold_score(
                d,
                alpha=alpha,
            )

        rows.append(row)

    return pd.DataFrame(rows)


def make_model_label(row, prefix="NN"):
    return (
        f"{prefix} {row['x_set_name']} "
        f"{row['variant']} "
        f"widths={row['widths']} "
        f"xi={row['xi_init']}"
    )


def savefig(fig, name):
    fig.savefig(
        IM_FOLDER / name,
        dpi=300,
        bbox_inches="tight",
    )


# %%
# Load and prepare the modeling data
print("Initial EGPD parameters")
print({"sigma": SIGMA_INIT, "kappa": KAPPA_INIT, "xi": XI_INIT})

df_raw = pd.read_csv(DOWNSCALING_TABLE, sep=";")
df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True)

print("Missing values in Y_obs:", df_raw["Y_obs"].isna().sum())
print("Proportion of Y_obs = 0:", (df_raw["Y_obs"] == 0).mean())

df_model, x_cols27, x_cols_dt0h, x_cols_all = prepare_modeling_dataframe(df_raw)

print("x_cols27:", len(x_cols27))
print("x_cols_dt0h:", len(x_cols_dt0h))
print("x_cols_all:", len(x_cols_all))
print("df_model shape:", df_model.shape)

single_cov_col = SINGLE_COV_COL

if single_cov_col not in df_model.columns:
    raise ValueError(f"{single_cov_col} not found in df_model columns.")

print("Single covariate used for GLM/GAM:", single_cov_col)

x_sets = make_covariate_sets(
    x_cols27=x_cols27,
    x_cols_dt0h=x_cols_dt0h,
    x_cols_all=x_cols_all,
)

print("Available NN covariate sets")

for name, cols in x_sets.items():
    print(f"{name:20s}: {len(cols)} variables")


# %%
# Create a blocked train-validation-test split
df_train_valid, df_test, test_split_info = make_train_valid_test_split(
    df=df_model,
    test_frac=0.10,
    block="7D",
    seed=2026,
)

cv_splits = make_blocked_cv_splits(
    df=df_train_valid,
    n_splits=3,
    block="7D",
    seed=1,
)

print("Train-validation period")
print(df_train_valid["time"].min(), "->", df_train_valid["time"].max())

print("Held-out test period")
print(df_test["time"].min(), "->", df_test["time"].max())

for sp in cv_splits:
    print(
        f"fold={sp['fold']} | "
        f"n_train={len(sp['train_idx'])} | "
        f"n_valid={len(sp['valid_idx'])} | "
        f"n_valid_blocks={len(sp['valid_blocks'])}"
    )


# %%
# Visual check of the temporal split
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df_model["time"],
    df_model["Y_obs"],
    alpha=0.18,
    label="All observations",
)

for sp in cv_splits:
    df_valid_fold = df_train_valid.loc[sp["valid_idx"]]

    ax.scatter(
        df_valid_fold["time"],
        df_valid_fold["Y_obs"],
        s=6,
        alpha=0.35,
        label=f"CV validation fold {sp['fold']}",
    )

ax.scatter(
    df_test["time"],
    df_test["Y_obs"],
    s=8,
    alpha=0.55,
    label="Test",
)

ax.set_xlabel("Time", size = 14)
ax.set_ylabel("Observed positive rainfall", size = 14)
ax.grid(True, alpha=0.3)
ax.legend(ncol=2)

plt.tight_layout()
plt.show()

savefig(fig, "cv_splits_with_heldout_test.png")


# %%
# Define a focused NN grid using only the both variant
nn_param_grid = {
    "variant": ["both"],
    "x_set_name": ["radar_time_space"],
    "widths": [
        (4, 2),
        (6, 3),
        (8, 4),
    ],
    "lr": [
        5e-4,
        1e-4,
    ],
    "weight_decay": [
        0.0,
    ],
    "batch_size": [
        64,
        128,
    ],
    "n_ep": [
        300,
    ],
    "sigma_init": [
        SIGMA_INIT,
    ],
    "kappa_init": [
        KAPPA_INIT,
    ],
    "xi_init": [
        0.20,
        XI_INIT,
        0.30,
        0.40,
    ],
    "censor_threshold": [
        CENSOR_THRESHOLD,
        0.40,
    ],
    "init_source": [
        "default",
        "gam",
    ],
}

# %%
# Tune NN configurations on one outer split
tuning_df, best_params_initial, best_history_initial = tune_nn_on_outer_train(
    df_model=df_train_valid,
    outer_split=cv_splits[0],
    x_sets=x_sets,
    param_grid=nn_param_grid,
    seed=1,
    device=None,
    single_cov_col=single_cov_col,
)

tuning_df.to_csv(OUT_TUNING, index=False)

print("Initial best NN parameters based on validation likelihood")
print(best_params_initial)

cols_show = [
    "x_set_name",
    "n_covariates",
    "variant",
    "widths",
    "init_source",
    "sigma_init",
    "kappa_init",
    "xi_init",
    "lr",
    "weight_decay",
    "batch_size",
    "train_loss",
    "valid_loss",
    "stopped_epoch",
]

cols_show = [c for c in cols_show if c in tuning_df.columns]

print(
    tuning_df
    .sort_values("valid_loss")
    .head(30)[cols_show]
    .to_string(index=False)
)


# %%
# Plot the tuning results for all NN configurations
plot_df = tuning_df.copy()
plot_df = plot_df.sort_values("valid_loss").reset_index(drop=True)
plot_df["config_rank"] = np.arange(1, len(plot_df) + 1)
plot_df["gap_valid_train"] = plot_df["valid_loss"] - plot_df["train_loss"]

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    plot_df["config_rank"],
    plot_df["train_loss"],
    marker="o",
    linewidth=1,
    markersize=3,
    label="Train NLL",
)

ax.plot(
    plot_df["config_rank"],
    plot_df["valid_loss"],
    marker="o",
    linewidth=1,
    markersize=3,
    label="Validation NLL",
)

ax.set_xlabel("Configuration rank sorted by validation NLL")
ax.set_ylabel("Negative log-likelihood")
ax.set_title("NN tuning across configurations")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

savefig(fig, "nn_tuning_train_valid_nll_all_configs.png")

#%%
def history_to_dataframe(history):
    if history is None:
        raise ValueError("No history was returned.")

    train_key = "train_nll" if "train_nll" in history else "train_epoch"
    valid_key = "valid_nll" if "valid_nll" in history else "val_epoch"

    return pd.DataFrame({
        "epoch": np.arange(1, len(history[train_key]) + 1),
        "train_nll": history[train_key],
        "valid_nll": history[valid_key],
    })


hist_df = history_to_dataframe(best_history_initial)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(hist_df["epoch"], hist_df["train_nll"], label="Train NLL")
ax.plot(hist_df["epoch"], hist_df["valid_nll"], label="Validation NLL")

best_epoch = hist_df["valid_nll"].idxmin() + 1
best_valid = hist_df["valid_nll"].min()

ax.scatter(
    best_epoch,
    best_valid,
    s=60,
    color="black",
    label="Best validation epoch",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Negative log-likelihood")
ax.set_title("Training history of the selected NN")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

savefig(fig, "selected_nn_training_history.png")

# %%
# Plot the best NN configurations with readable labels
top_k_plot = min(30, len(tuning_df))

plot_top = (
    tuning_df
    .sort_values("valid_loss")
    .head(top_k_plot)
    .reset_index(drop=True)
)

plot_top["config_rank"] = np.arange(1, len(plot_top) + 1)
plot_top["gap_valid_train"] = plot_top["valid_loss"] - plot_top["train_loss"]

plot_top["label"] = (
    plot_top["x_set_name"].astype(str)
    + "\n"
    + plot_top["widths"].astype(str)
    + "\n"
    + "xi="
    + plot_top["xi_init"].astype(str)
    + "\n"
    + plot_top["init_source"].astype(str)
)

fig, ax = plt.subplots(figsize=(15, 6))

x = np.arange(len(plot_top))
width = 0.35

ax.bar(
    x - width / 2,
    plot_top["train_loss"],
    width,
    label="Train NLL",
)

ax.bar(
    x + width / 2,
    plot_top["valid_loss"],
    width,
    label="Validation NLL",
)

ax.set_xticks(x)
ax.set_xticklabels(plot_top["label"], rotation=45, ha="right")
ax.set_ylabel("Negative log-likelihood")
ax.set_title("Best NN configurations according to validation likelihood")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

savefig(fig, "nn_tuning_nll_barplot_top_configs.png")


# %%
# Plot the generalization gap for the best NN configurations

fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(
    plot_top["config_rank"],
    plot_top["gap_valid_train"],
)

ax.axhline(0.0, color="black", linewidth=0.8)
ax.set_xlabel("Configuration rank")
ax.set_ylabel("Validation NLL - Train NLL")
ax.set_title("Generalization gap for the best NN configurations")
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

savefig(fig, "nn_tuning_nll_gap_top_configs.png")


# %%
# Rerank the best likelihood-based NN configurations using tail-aware scores

rerank_df, best_params, best_history = rerank_top_nn_configs(
    df_model=df_train_valid,
    outer_split=cv_splits[0],
    x_sets=x_sets,
    tuning_df=tuning_df,
    top_k=15,
    seed=1,
    device=None,
)

rerank_df.to_csv(OUT_RERANK, index=False)
pd.DataFrame([best_params]).to_csv(OUT_BEST_PARAMS, index=False)

print("Best NN parameters after reranking")
print(best_params)

print(
    rerank_df
    .head(20)
    .to_string(index=False)
)


# %%
# Plot the reranked NN configurations

plot_rerank = rerank_df.copy()

sort_rerank_cols = [
    c for c in [
        "twcrps_sum",
        "valid_loss",
        "crps_mean",
        "smad",
    ]
    if c in plot_rerank.columns
]

plot_rerank = (
    plot_rerank
    .sort_values(sort_rerank_cols, na_position="last")
    .reset_index(drop=True)
)

plot_rerank["rank"] = np.arange(1, len(plot_rerank) + 1)
plot_rerank["gap_valid_train"] = plot_rerank["valid_loss"] - plot_rerank["train_loss"]

plot_rerank["label"] = (
    plot_rerank["x_set_name"].astype(str)
    + "\n"
    + plot_rerank["widths"].astype(str)
    + "\n"
    + "xi="
    + plot_rerank["xi_init"].astype(str)
)

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(plot_rerank))
width = 0.35

ax.bar(
    x - width / 2,
    plot_rerank["train_loss"],
    width,
    label="Train NLL",
)

ax.bar(
    x + width / 2,
    plot_rerank["valid_loss"],
    width,
    label="Validation NLL",
)

ax.set_xticks(x)
ax.set_xticklabels(plot_rerank["label"], rotation=45, ha="right")
ax.set_ylabel("Negative log-likelihood")
ax.set_title("Reranked NN configurations: likelihood comparison")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

savefig(fig, "nn_rerank_train_valid_nll.png")


# %%
# Plot reranking scores together after normalization

score_cols = [
    c for c in [
        "twcrps_sum",
        "crps_mean",
        "smad",
    ]
    if c in plot_rerank.columns
]

norm_df = plot_rerank[["rank"] + score_cols].copy()

for col in score_cols:
    col_min = norm_df[col].min()
    col_max = norm_df[col].max()

    if col_max > col_min:
        norm_df[f"{col}_norm"] = (norm_df[col] - col_min) / (col_max - col_min)
    else:
        norm_df[f"{col}_norm"] = 0.0

fig, ax = plt.subplots(figsize=(12, 5))

for col in score_cols:
    ax.plot(
        norm_df["rank"],
        norm_df[f"{col}_norm"],
        marker="o",
        label=col,
    )

ax.set_xlabel("Reranked configuration")
ax.set_ylabel("Normalized score")
ax.set_title("Normalized NN reranking scores")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

savefig(fig, "nn_rerank_scores_normalized.png")


# %%
# Compare validation likelihood and tail-weighted CRPS

if "twcrps_sum" in plot_rerank.columns:
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        plot_rerank["valid_loss"],
        plot_rerank["twcrps_sum"],
        s=60,
    )

    for _, row in plot_rerank.iterrows():
        ax.annotate(
            str(int(row["rank"])),
            (row["valid_loss"], row["twcrps_sum"]),
            textcoords="offset points",
            xytext=(5, 5),
        )

    ax.set_xlabel("Validation NLL")
    ax.set_ylabel("Tail-weighted CRPS")
    ax.set_title("Validation likelihood versus tail score")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    savefig(fig, "nn_rerank_valid_loss_vs_twcrps.png")


# %%
# Select the best NN configurations to inspect on the held-out test set
selection_score = "twcrps_sum" if "twcrps_sum" in rerank_df.columns else "valid_loss"
top_k_final = min(5, len(rerank_df))

top_nn_configs = (
    rerank_df
    .sort_values(selection_score)
    .head(top_k_final)
    .reset_index(drop=True)
)

print("Top NN configurations selected according to:", selection_score)

cols_top = [
    "x_set_name",
    "n_covariates",
    "variant",
    "widths",
    "init_source",
    "sigma_init",
    "kappa_init",
    "xi_init",
    "lr",
    "weight_decay",
    "batch_size",
    "train_loss",
    "valid_loss",
    "twcrps_sum",
    "crps_mean",
    "smad",
]

cols_top = [c for c in cols_top if c in top_nn_configs.columns]

print(top_nn_configs[cols_top].to_string(index=False))


# %%
# Refit the best NN configurations and evaluate them on train-validation and test samples
nn_train_rows = []
nn_test_rows = []
nn_train_preds = []
nn_test_preds = []

for i, cfg_row in top_nn_configs.iterrows():
    params = cfg_row.to_dict()
    params["n_ep"] = 300

    label = make_model_label(params, prefix=f"NN{i + 1}")

    print("Fitting:", label)

    row_train, pred_train = fit_predict_nn_test(
        df_train_valid=df_train_valid,
        df_test=df_train_valid,
        x_sets=x_sets,
        best_params=params,
        seed=123 + i,
        device=None,
    )

    row_train["sample"] = "train_valid"
    row_train["model"] = label
    pred_train["sample"] = "train_valid"
    pred_train["model"] = label

    nn_train_rows.append(row_train)
    nn_train_preds.append(pred_train)

    row_test, pred_test = fit_predict_nn_test(
        df_train_valid=df_train_valid,
        df_test=df_test,
        x_sets=x_sets,
        best_params=params,
        seed=123 + i,
        device=None,
    )

    row_test["sample"] = "test"
    row_test["model"] = label
    pred_test["sample"] = "test"
    pred_test["model"] = label

    nn_test_rows.append(row_test)
    nn_test_preds.append(pred_test)


# %%
# Summarize the best NN configurations on train-validation and test samples

nn_train_res = pd.DataFrame(nn_train_rows)
nn_test_res = pd.DataFrame(nn_test_rows)

nn_train_test_res = pd.concat(
    [nn_train_res, nn_test_res],
    ignore_index=True,
    sort=False,
)

sort_cols = [
    c for c in [
        "sample",
        "twcrps_sum",
        "test_loss",
        "crps_mean",
        "smad",
    ]
    if c in nn_train_test_res.columns
]

nn_train_test_res = (
    nn_train_test_res
    .sort_values(sort_cols, na_position="last")
    .reset_index(drop=True)
)

print("NN train-test comparison")
print(nn_train_test_res.to_string(index=False))

nn_train_test_res.to_csv(
    IM_FOLDER / "nn_top_configs_train_test_scores.csv",
    index=False,
)


# %%
# Compute alpha-sensitive tail scores for the best NN configurations on test data

pred_nn_test = pd.concat(
    nn_test_preds,
    ignore_index=True,
    sort=False,
)

pred_nn_test = add_prediction_quantities(
    pred_nn_test,
    quantiles=(
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.75,
        0.8,
        0.9,
        0.95,
        0.99,
        0.995,
        0.999,
    ),
)

pred_nn_test.to_csv(
    IM_FOLDER / "nn_top_configs_test_predictions.csv",
    index=False,
)

nn_alpha_scores = add_twcrps_alpha_scores(
    pred_nn_test,
    model_col="model",
    alphas=(0.0, 1.0, 2.0),
)

nn_alpha_scores.to_csv(
    IM_FOLDER / "nn_top_configs_twcrps_alpha_scores.csv",
    index=False,
)

print("Alpha-sensitive tail scores for the best NN configurations")
print(nn_alpha_scores.to_string(index=False))


# %%
# Plot alpha-sensitive tail scores for the best NN configurations

plot_alpha = nn_alpha_scores.copy()
plot_alpha = plot_alpha.sort_values("twcrps_alpha_1").reset_index(drop=True)
plot_alpha["rank"] = np.arange(1, len(plot_alpha) + 1)

fig, ax = plt.subplots(figsize=(10, 5))

for alpha in (0.0, 1.0, 2.0):
    col = f"twcrps_alpha_{alpha:g}"

    if col in plot_alpha.columns:
        ax.plot(
            plot_alpha["rank"],
            plot_alpha[col],
            marker="o",
            label=rf"$\alpha={alpha:g}$",
        )

ax.set_xlabel("NN configuration rank")
ax.set_ylabel("Tail-weighted threshold score")
ax.set_title("Sensitivity of NN ranking to the tail weight")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

savefig(fig, "nn_top_configs_twcrps_alpha_sensitivity.png")


# %%
# Diagnostic plots for the best NN configurations on the test set

model_order_nn = pred_nn_test["model"].unique().tolist()

plot_exponential_qq(
    pred_nn_test,
    model_order=model_order_nn,
    p_low=0.50,
    p_high=0.999,
    save_name="qq_exponential_top_nn_configs_test",
)

plot_pit_histograms(
    pred_nn_test,
    model_order=model_order_nn,
    n_bins=20,
)

plot_quantile_calibration(
    pred_nn_test,
    model_order=model_order_nn,
    q_levels=(0.75, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999),
    save_name="quantile_calibration_top_nn_configs_test",
)

plot_tail_exceedance_calibration(
    pred_nn_test,
    model_order=model_order_nn,
    thresholds=(0.5, 1.0, 2.0, 5.0, 10.0),
    save_name="tail_exceedance_top_nn_configs_test",
)


# %%
# Choose the final NN configuration for the model comparison

best_params_final = top_nn_configs.iloc[0].to_dict()
best_params_final["n_ep"] = 300

best_nn_label = (
    f"NN {best_params_final['x_set_name']} "
    f"both widths={best_params_final['widths']}"
)

print("Final NN selected for comparison")
print(best_nn_label)
print(best_params_final)


# %%
# Compare the final NN with stationary, GLM both, and GAM both using blocked CV

all_results = []

print("Running stationary EGPD on blocked CV")

res_stationary = evaluate_stationary_candidate(
    df_model=df_train_valid,
    splits=cv_splits,
    sigma_init=SIGMA_INIT,
    kappa_init=KAPPA_INIT,
    xi_init=XI_INIT,
    censor_threshold=CENSOR_THRESHOLD,
    fix_xi=True,
)

all_results.append(res_stationary)

print("Running GLM both on blocked CV")

res_glm = evaluate_single_covariate_model(
    df_model=df_train_valid,
    splits=cv_splits,
    model_type="glm",
    variant="both",
    covariate_col=single_cov_col,
    sigma_init=SIGMA_INIT,
    kappa_init=KAPPA_INIT,
    xi_init=XI_INIT,
    censor_threshold=CENSOR_THRESHOLD,
)

all_results.append(res_glm)

print("Running GAM both on blocked CV")

res_gam = evaluate_single_covariate_model(
    df_model=df_train_valid,
    splits=cv_splits,
    model_type="gam",
    variant="both",
    covariate_col=single_cov_col,
    sigma_init=SIGMA_INIT,
    kappa_init=KAPPA_INIT,
    xi_init=XI_INIT,
    censor_threshold=CENSOR_THRESHOLD,
)

all_results.append(res_gam)

print("Running final NN both on blocked CV")

res_nn_final = evaluate_fixed_nn_model(
    df_model=df_train_valid,
    splits=cv_splits,
    x_sets=x_sets,
    best_params=best_params_final,
    seed=1,
    device=None,
)

all_results.append(res_nn_final)

comparison_res = pd.concat(
    all_results,
    ignore_index=True,
    sort=False,
)

comparison_res.to_csv(
    OUT_COMPARISON,
    index=False,
)

summary = summarize_model_comparison(comparison_res)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)

print("Blocked CV comparison summary")
print(summary.to_string(index=False))


# %%
# Export deltas to the best model for easier interpretation

cols_delta = [
    "model_family",
    "model_type",
    "variant",
    "covariate",
    "n_folds",
    "valid_loss_mean",
    "valid_loss_sum_mean",
    "twcrps_sum_mean",
    "crps_mean",
    "smad_mean",
    "smad_original_mean",
    "err95_mean",
    "err99_mean",
    "valid_loss_delta",
    "valid_loss_sum_delta",
    "twcrps_sum_delta",
    "crps_delta",
    "smad_delta",
    "smad_original_delta",
    "err95_delta",
    "err99_delta",
]

cols_delta = [c for c in cols_delta if c in summary.columns]
summary_delta = summary[cols_delta].copy()

summary_delta.to_csv(
    OUT_SUMMARY_DELTA,
    index=False,
)

print("Delta to the best model")
print(summary_delta.to_string(index=False))


# %%
# Plot the CV model comparison

plot_summary = summary.copy()

if "twcrps_sum_mean" in plot_summary.columns:
    plot_summary = plot_summary.sort_values("twcrps_sum_mean").reset_index(drop=True)

label_cols = [c for c in ["model_type", "variant", "covariate"] if c in plot_summary.columns]

plot_summary["label"] = (
    plot_summary[label_cols]
    .astype(str)
    .agg(" ".join, axis=1)
    .str.replace("nan", "", regex=False)
    .str.strip()
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    plot_summary["label"],
    plot_summary["twcrps_sum_mean"],
)

ax.set_ylabel("Mean tail-weighted CRPS")
ax.set_title("Blocked CV comparison")
ax.grid(True, axis="y", alpha=0.3)
ax.tick_params(axis="x", rotation=35)

plt.tight_layout()
plt.show()

savefig(fig, "cv_model_comparison_twcrps.png")


# %%
# Fit the final models on train-validation and evaluate them on the held-out test set

test_rows = []
test_preds = []

print("Test evaluation: stationary EGPD")

row, pred = fit_predict_stationary_test(
    df_train_valid=df_train_valid,
    df_test=df_test,
)

row["model"] = "Stationary EGPD"
pred["model"] = "Stationary EGPD"

test_rows.append(row)
test_preds.append(pred)

print("Test evaluation: GLM both")

row, pred = fit_predict_regression_test(
    df_train_valid=df_train_valid,
    df_test=df_test,
    model_type="glm",
    variant="both",
    covariate_col=single_cov_col,
)

row["model"] = "GLM both"
pred["model"] = "GLM both"

test_rows.append(row)
test_preds.append(pred)

print("Test evaluation: GAM both")

row, pred = fit_predict_regression_test(
    df_train_valid=df_train_valid,
    df_test=df_test,
    model_type="gam",
    variant="both",
    covariate_col=single_cov_col,
)

row["model"] = "GAM both"
pred["model"] = "GAM both"

test_rows.append(row)
test_preds.append(pred)

print("Test evaluation: final NN both")

row, pred = fit_predict_nn_test(
    df_train_valid=df_train_valid,
    df_test=df_test,
    x_sets=x_sets,
    best_params=best_params_final,
    seed=123,
    device=None,
)

row["model"] = best_nn_label
pred["model"] = best_nn_label

test_rows.append(row)
test_preds.append(pred)

test_res = pd.DataFrame(test_rows)

sort_test_cols = [
    c for c in [
        "twcrps_sum",
        "test_loss",
        "crps_mean",
        "smad",
    ]
    if c in test_res.columns
]

test_res = (
    test_res
    .sort_values(sort_test_cols, na_position="last")
    .reset_index(drop=True)
)

test_res.to_csv(
    OUT_TEST,
    index=False,
)

print("Held-out test comparison")
print(test_res.to_string(index=False))


# %%
# Build final prediction table and compute alpha-sensitive tail scores

pred_test_all = pd.concat(
    test_preds,
    ignore_index=True,
    sort=False,
)

pred_test_all = add_prediction_quantities(
    pred_test_all,
    quantiles=(
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.75,
        0.8,
        0.9,
        0.95,
        0.99,
        0.995,
        0.999,
    ),
)

pred_test_all.to_csv(
    OUT_TEST_PRED,
    index=False,
)

final_alpha_scores = add_twcrps_alpha_scores(
    pred_test_all,
    model_col="model",
    alphas=(0.0, 1.0, 2.0),
)

final_alpha_scores.to_csv(
    IM_FOLDER / "final_model_comparison_twcrps_alpha_scores.csv",
    index=False,
)

print("Alpha-sensitive tail scores for final model comparison")
print(final_alpha_scores.to_string(index=False))


# %%
# Plot the alpha-sensitive scores for the final model comparison

plot_alpha_final = final_alpha_scores.copy()
plot_alpha_final = plot_alpha_final.sort_values("twcrps_alpha_1").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(plot_alpha_final))
width = 0.25

for j, alpha in enumerate((0.0, 1.0, 2.0)):
    col = f"twcrps_alpha_{alpha:g}"

    if col in plot_alpha_final.columns:
        ax.bar(
            x + (j - 1) * width,
            plot_alpha_final[col],
            width,
            label=rf"$\alpha={alpha:g}$",
        )

ax.set_xticks(x)
ax.set_xticklabels(plot_alpha_final["model"], rotation=30, ha="right")
ax.set_ylabel("Tail-weighted threshold score")
ax.set_title("Sensitivity of final model comparison to the tail weight")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

savefig(fig, "final_model_comparison_twcrps_alpha_sensitivity.png")


# %%
# Run final distributional diagnostics

model_order = [
    m for m in [
        "Stationary EGPD",
        "GLM both",
        "GAM both",
        best_nn_label,
    ]
    if m in pred_test_all["model"].unique()
]

print("Model order for diagnostics")
print(model_order)

plot_exponential_qq(
    pred_test_all,
    model_order=model_order,
    p_low=0.50,
    p_high=0.999,
    save_name="qq_exponential_final_model_comparison_test",
)

plot_pit_histograms(
    pred_test_all,
    model_order=model_order,
    n_bins=20,
)

plot_quantile_calibration(
    pred_test_all,
    model_order=model_order,
    q_levels=(0.75, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999),
    save_name="quantile_calibration_final_model_comparison_test",
)

plot_tail_exceedance_calibration(
    pred_test_all,
    model_order=model_order,
    thresholds=(0.5, 1.0, 2.0, 5.0, 10.0),
    save_name="tail_exceedance_final_model_comparison_test",
)


# %%
# Compare observed and simulated test distributions for the final NN

rng = np.random.default_rng(123)

d_nn = pred_test_all[pred_test_all["model"] == best_nn_label].copy()

u = rng.uniform(size=len(d_nn))

d_nn["Y_sim"] = egpd_quantile(
    u,
    d_nn["sigma"].to_numpy(),
    d_nn["kappa"].to_numpy(),
    d_nn["xi"].to_numpy(),
)

upper = np.nanquantile(
    np.concatenate(
        [
            d_nn["Y_obs"].to_numpy(),
            d_nn["Y_sim"].to_numpy(),
        ]
    ),
    0.995,
)

bins = np.linspace(0.0, upper, 60)

fig, ax = plt.subplots(figsize=(7, 5))

ax.hist(
    d_nn["Y_obs"],
    bins=bins,
    density=True,
    alpha=0.45,
    label="Observed gauge rainfall",
)

ax.hist(
    d_nn["Y_sim"],
    bins=bins,
    density=True,
    alpha=0.45,
    label="Simulated from conditional EGPD",
)

ax.set_xlabel("Positive 5-min rainfall")
ax.set_ylabel("Density")
ax.set_title("Observed versus simulated local positive rainfall")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

savefig(fig, "final_nn_observed_vs_simulated_distribution.png")


# %%
# Plot a short predictive time series for the final NN

d_ts = d_nn.sort_values("time").copy()

if "station" in d_ts.columns:
    station_counts = d_ts["station"].value_counts()
    station_to_plot = station_counts.index[0]
    d_ts = d_ts[d_ts["station"] == station_to_plot].copy()
    title_suffix = f" at station {station_to_plot}"
else:
    title_suffix = ""

d_ts = d_ts.head(500)

fig, ax = plt.subplots(figsize=(13, 5))

ax.plot(
    d_ts["time"],
    d_ts["Y_obs"],
    color="black",
    linewidth=1,
    label="Observed gauge rainfall",
)

if "q0.5" in d_ts.columns:
    ax.plot(
        d_ts["time"],
        d_ts["q0.5"],
        linewidth=1.5,
        label="Predictive median",
    )

if {"q0.75", "q0.95"}.issubset(d_ts.columns):
    ax.fill_between(
        d_ts["time"],
        d_ts["q0.75"],
        d_ts["q0.95"],
        alpha=0.25,
        label="75–95% predictive range",
    )

if {"q0.95", "q0.99"}.issubset(d_ts.columns):
    ax.fill_between(
        d_ts["time"],
        d_ts["q0.95"],
        d_ts["q0.99"],
        alpha=0.15,
        label="95–99% predictive range",
    )

ax.set_xlabel("Time")
ax.set_ylabel("Positive 5-min rainfall")
ax.set_title(f"Conditional predictive distribution{title_suffix}")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

savefig(fig, "final_nn_predictive_time_series.png")


# %%
# Check the implied censoring probability at the censoring threshold

print("Censoring probability check at threshold:", CENSOR_THRESHOLD)

for model in model_order:
    d = pred_test_all[pred_test_all["model"] == model]

    fc = egpd_cdf(
        CENSOR_THRESHOLD,
        d["sigma"].to_numpy(),
        d["kappa"].to_numpy(),
        d["xi"].to_numpy(),
    )

    print(
        model,
        "| median F(c) =",
        np.median(fc),
        "| mean F(c) =",
        np.mean(fc),
    )


#%%
for station in pred_test_all["station"].unique():
    d_station = pred_test_all[pred_test_all["station"] == station]

    plot_exponential_qq(
        d_station,
        model_order=[best_nn_label],
        p_low=0.50,
        p_high=0.999,
        save_name=f"qq_exponential_station_{station}",
    )

#%%
d = pred_test_all[pred_test_all["model"] == best_nn_label].copy()
d["date"] = d["time"].dt.floor("D")

daily = (
    d.groupby("date")
    .agg(
        sigma_mean=("sigma", "mean"),
        kappa_mean=("kappa", "mean"),
        q95_mean=("q0.95", "mean"),
        q99_mean=("q0.99", "mean"),
        y_mean=("Y_obs", "mean"),
    )
    .reset_index()
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(daily["date"], daily["sigma_mean"], label=r"Mean $\sigma(s,t)$")
ax.plot(daily["date"], daily["q95_mean"], label=r"Mean $q_{0.95}(s,t)$")
ax.plot(daily["date"], daily["q99_mean"], label=r"Mean $q_{0.99}(s,t)$")

ax.set_xlabel("Time")
ax.set_ylabel("Conditional rainfall intensity metric")
ax.set_title("Time evolution of spatially averaged conditional rainfall metrics")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()