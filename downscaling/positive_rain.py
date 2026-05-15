# %%
import os
from pathlib import Path
import sys

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
    TUNING_PRESET,
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
# Load and prepare data
print("Initial parameters:")
print({"sigma": SIGMA_INIT, "kappa": KAPPA_INIT, "xi": XI_INIT})

df_raw = pd.read_csv(DOWNSCALING_TABLE, sep=";")
df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True)

print("Missing values in Y_obs:", df_raw["Y_obs"].isna().sum())
print("Proportion of Y_obs = 0:", (df_raw["Y_obs"] == 0).mean())

df_model, x_cols27, x_cols_dt0h, x_cols_all = prepare_modeling_dataframe(df_raw)

print("x_cols27 len:", len(x_cols27))
print("x_cols_dt0h len:", len(x_cols_dt0h))
print("x_cols_all len:", len(x_cols_all))
print("df_model shape:", df_model.shape)

single_cov_col = SINGLE_COV_COL
if single_cov_col not in df_model.columns:
    raise ValueError(f"{single_cov_col} not found in df_model columns.")

print(f"Using single covariate: {single_cov_col}")

x_sets = make_covariate_sets(
    x_cols27=x_cols27,
    x_cols_dt0h=x_cols_dt0h,
    x_cols_all=x_cols_all,
)

print("\nAvailable NN covariate sets:")
for name, cols in x_sets.items():
    print(f"{name:20s}: {len(cols)} variables")

# %%
# Train / valid / test split
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

print("\nTrain/valid period:")
print(df_train_valid["time"].min(), "->", df_train_valid["time"].max())

print("\nTest period:")
print(df_test["time"].min(), "->", df_test["time"].max())

for sp in cv_splits:
    print(
        f"fold={sp['fold']} | "
        f"n_train={len(sp['train_idx'])} | "
        f"n_valid={len(sp['valid_idx'])} | "
        f"n_valid_blocks={len(sp['valid_blocks'])}"
    )

# %%
# Plot split check

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
        label=f"CV valid fold {sp['fold']}",
    )

ax.scatter(
    df_test["time"],
    df_test["Y_obs"],
    s=8,
    alpha=0.55,
    label="Held-out test",
)

ax.set_xlabel("Time")
ax.set_ylabel("Y_obs")
ax.grid(True, alpha=0.3)
ax.legend(ncol=2)

plt.tight_layout()
plt.show()

fig.savefig(
    IM_FOLDER / "cv_splits_with_heldout_test.png",
    dpi=300,
    bbox_inches="tight",
)

# %%
#  NN hyperparameter grid
nn_param_grid = {
        "variant": ["both"],
        "x_set_name": ["radar_time_space"],
        "widths": [
            # (4, 2),
            (6, 3),
            (8, 4),
            (12, 6),
            (8, 4, 2),
            # (12, 6, 3),
            # (16, 8, 4),
        ],
        "lr": [1e-3],
        "weight_decay": [0.0],
        "batch_size": [128],
        "n_ep": [100],
        "sigma_init": [SIGMA_INIT],
        "kappa_init": [KAPPA_INIT],
        "xi_init": [0.20, XI_INIT, 0.30],
        "censor_threshold": [0.22, 0.40],
        "init_source": ["default"], #, "gam"
}

# %%
#  NN tuning on one outer split
tuning_df, best_params_initial = tune_nn_on_outer_train(
    df_model=df_train_valid,
    outer_split=cv_splits[0],
    x_sets=x_sets,
    param_grid=nn_param_grid,
    seed=1,
    device=None,
    single_cov_col=single_cov_col
)

tuning_df.to_csv(OUT_TUNING, index=False)

print("\nInitial best params from tuning:")
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
# NN tuning plots: train/valid likelihoods

plot_df = tuning_df.copy()
plot_df = plot_df.sort_values("valid_loss").reset_index(drop=True)
plot_df["config_rank"] = np.arange(1, len(plot_df) + 1)

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

ax.set_xlabel("Configuration rank, sorted by validation NLL")
ax.set_ylabel("Negative log-likelihood")
ax.set_title("NN tuning: train vs validation likelihood")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

fig.savefig(
    IM_FOLDER / "nn_tuning_train_valid_nll_all_configs.png",
    dpi=300,
    bbox_inches="tight",
)

# %%
TOP_K_PLOT = min(30, len(tuning_df))

plot_top = (
    tuning_df
    .sort_values("valid_loss")
    .head(TOP_K_PLOT)
    .reset_index(drop=True)
)

plot_top["config_rank"] = np.arange(1, len(plot_top) + 1)
plot_top["gap_valid_train"] = plot_top["valid_loss"] - plot_top["train_loss"]

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    plot_top["config_rank"],
    plot_top["train_loss"],
    marker="o",
    label="Train NLL",
)

ax.plot(
    plot_top["config_rank"],
    plot_top["valid_loss"],
    marker="o",
    label="Validation NLL",
)

ax.set_xlabel(f"Top {TOP_K_PLOT} configurations")
ax.set_ylabel("Negative log-likelihood")
ax.set_title(f"Train vs validation NLL for top {TOP_K_PLOT} NN configurations")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

fig.savefig(
    IM_FOLDER / "nn_tuning_train_valid_nll_top_configs.png",
    dpi=300,
    bbox_inches="tight",
)



# %%
# Rerank top NN configs using tail-informative metrics/scores

rerank_df, best_params = rerank_top_nn_configs(
    df_model=df_train_valid,
    outer_split=cv_splits[0],
    x_sets=x_sets,
    tuning_df=tuning_df,
    top_k=10,
    seed=1,
    device=None,
)

rerank_df.to_csv(OUT_RERANK, index=False)
pd.DataFrame([best_params]).to_csv(OUT_BEST_PARAMS, index=False)

print("\nBest params after reranking:")
print(best_params)

print(
    rerank_df
    .head(20)
    .to_string(index=False)
)

#%%
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(tuning_df["train_loss"], tuning_df["valid_loss"], alpha=0.7)

ax.set_xlabel("Train NLL")
ax.set_ylabel("Validation NLL")
ax.set_title("Train vs validation NLL across NN configs")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# %%
# Plots after reranking: likelihoods and scores

plot_rerank = rerank_df.copy().reset_index(drop=True)

# Ensure final order is according to the final selection criterion
plot_rerank = plot_rerank.sort_values(
    [
        "twcrps_sum",
        "valid_loss",
        "crps_mean",
        "smad",
    ],
    na_position="last",
).reset_index(drop=True)

plot_rerank["rank"] = np.arange(1, len(plot_rerank) + 1)

plot_rerank["label"] = (
    plot_rerank["x_set_name"].astype(str)
    + "\n" + plot_rerank["variant"].astype(str)
    + "\n" + plot_rerank["widths"].astype(str)
    + "\nxi=" + plot_rerank["xi_init"].astype(str)
)

# %%
# Train vs validation NLL after reranking

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
ax.set_title("Top NN configurations after reranking: train vs validation NLL")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

fig.savefig(
    IM_FOLDER / "nn_rerank_train_valid_nll.png",
    dpi=300,
    bbox_inches="tight",
)

# %%
# Generalization gap

plot_rerank["gap_valid_train"] = (
    plot_rerank["valid_loss"] - plot_rerank["train_loss"]
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(
    plot_rerank["rank"],
    plot_rerank["gap_valid_train"],
)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Reranked configuration")
ax.set_ylabel("Validation NLL - Train NLL")
ax.set_title("Generalization gap after NN reranking")
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

fig.savefig(
    IM_FOLDER / "nn_rerank_nll_gap.png",
    dpi=300,
    bbox_inches="tight",
)

# %%
# Main scores after reranking

score_cols = [
    "twcrps_sum",
    "crps_mean",
    "smad",
]

score_cols = [c for c in score_cols if c in plot_rerank.columns]

for score in score_cols:
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(
        plot_rerank["rank"],
        plot_rerank[score],
    )

    ax.set_xlabel("Reranked configuration")
    ax.set_ylabel(score)
    ax.set_title(f"NN reranking score: {score}")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    fig.savefig(
        IM_FOLDER / f"nn_rerank_{score}.png",
        dpi=300,
        bbox_inches="tight",
    )

# %%
# Combined normalized score plot

norm_df = plot_rerank[["rank"] + score_cols].copy()

for c in score_cols:
    c_min = norm_df[c].min()
    c_max = norm_df[c].max()
    if c_max > c_min:
        norm_df[c + "_norm"] = (norm_df[c] - c_min) / (c_max - c_min)
    else:
        norm_df[c + "_norm"] = 0.0

fig, ax = plt.subplots(figsize=(12, 5))

for c in score_cols:
    ax.plot(
        norm_df["rank"],
        norm_df[c + "_norm"],
        marker="o",
        label=c,
    )

ax.set_xlabel("Reranked configuration")
ax.set_ylabel("Normalized score, lower is better")
ax.set_title("Normalized comparison of NN reranking scores")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

fig.savefig(
    IM_FOLDER / "nn_rerank_scores_normalized.png",
    dpi=300,
    bbox_inches="tight",
)

# %%
# Relationship between validation NLL and tail score

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
    ax.set_ylabel("twCRPS")
    ax.set_title("Validation likelihood vs tail-weighted CRPS")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    fig.savefig(
        IM_FOLDER / "nn_rerank_valid_loss_vs_twcrps.png",
        dpi=300,
        bbox_inches="tight",
    )

# %%
# Choose best configs using twCRPS
selection_score = (
    "twcrps_sum"
    if "twcrps_sum" in rerank_df.columns
    else "valid_loss"
)

TOP_K_FINAL = min(5, len(rerank_df))

top_nn_configs = (
    rerank_df
    .sort_values(selection_score)
    .head(TOP_K_FINAL)
    .reset_index(drop=True)
)

print(f"\nTop NN configs selected according to: {selection_score}")

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
# Evaluate several best NN configs on train_valid and test

nn_train_rows = []
nn_test_rows = []
nn_train_preds = []
nn_test_preds = []

for i, cfg_row in top_nn_configs.iterrows():
    params = cfg_row.to_dict()
    params["n_ep"] = 300

    label = (
        f"NN{i+1} {params['x_set_name']} "
        f"{params['variant']} widths={params['widths']} "
        f"xi={params['xi_init']}"
    )

    print("\nFitting:", label)

    # Apparent performance on train_valid
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

    # Independent held-out test performance
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

print("\nNN train/test comparison:")
print(nn_train_test_res.to_string(index=False))

nn_train_test_res.to_csv(
    IM_FOLDER / "nn_top_configs_train_test_scores.csv",
    index=False,
)

# %%
# ============================================================
# 8. Diagnostics for top NN configs on test set
# ============================================================

pred_nn_test = pd.concat(
    nn_test_preds,
    ignore_index=True,
    sort=False,
)

pred_nn_test = add_prediction_quantities(
    pred_nn_test,
    quantiles=(
        0.3, 0.4, 0.5, 0.6, 0.7,
        0.75, 0.8, 0.9, 0.95,
        0.99, 0.995, 0.999,
    ),
)

pred_nn_test.to_csv(
    IM_FOLDER / "nn_top_configs_test_predictions.csv",
    index=False,
)

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
# ============================================================
# 9. Choose final NN model for comparison with simpler models
# ============================================================

best_params_final = top_nn_configs.iloc[0].to_dict()
best_params_final["n_ep"] = 300

best_nn_label = (
    f"NN {best_params_final['x_set_name']} "
    f"{best_params_final['variant']} "
    f"widths={best_params_final['widths']}"
)

print("\nFinal NN selected for comparison:")
print(best_nn_label)
print(best_params_final)

# %%
# ============================================================
# 10. CV comparison: final NN vs stationary / GLM / GAM
# ============================================================

all_results = []

print("\nRunning stationary model on train_valid CV")

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

for variant in ["both", "sigma_only", "kappa_only"]:
    print(f"\nRunning GLM 1cov on train_valid CV - variant={variant}")

    res_glm = evaluate_single_covariate_model(
        df_model=df_train_valid,
        splits=cv_splits,
        model_type="glm",
        variant=variant,
        covariate_col=single_cov_col,
        sigma_init=SIGMA_INIT,
        kappa_init=KAPPA_INIT,
        xi_init=XI_INIT,
        censor_threshold=CENSOR_THRESHOLD,
    )

    all_results.append(res_glm)

    print(f"\nRunning GAM 1cov on train_valid CV - variant={variant}")

    res_gam = evaluate_single_covariate_model(
        df_model=df_train_valid,
        splits=cv_splits,
        model_type="gam",
        variant=variant,
        covariate_col=single_cov_col,
        sigma_init=SIGMA_INIT,
        kappa_init=KAPPA_INIT,
        xi_init=XI_INIT,
        censor_threshold=CENSOR_THRESHOLD,
    )

    all_results.append(res_gam)

print("\nRunning final NN on all CV folds")

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

print("\nCV comparison summary:")
print(summary.to_string(index=False))

# %%
# Delta to best model

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

print("\nDelta to best model:")
print(summary_delta.to_string(index=False))

# %%
# ============================================================
# 11. Held-out test comparison:
#     final NN vs stationary / GLM / GAM
# ============================================================

test_rows = []
test_preds = []

print("\nTest: stationary model")

row, pred = fit_predict_stationary_test(
    df_train_valid=df_train_valid,
    df_test=df_test,
)

test_rows.append(row)
test_preds.append(pred)

for variant in ["both", "sigma_only", "kappa_only"]:
    print(f"\nTest: GLM variant={variant}")

    row, pred = fit_predict_regression_test(
        df_train_valid=df_train_valid,
        df_test=df_test,
        model_type="glm",
        variant=variant,
        covariate_col=single_cov_col,
    )

    test_rows.append(row)
    test_preds.append(pred)

    print(f"\nTest: GAM variant={variant}")

    row, pred = fit_predict_regression_test(
        df_train_valid=df_train_valid,
        df_test=df_test,
        model_type="gam",
        variant=variant,
        covariate_col=single_cov_col,
    )

    test_rows.append(row)
    test_preds.append(pred)

print("\nTest: final NN")

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

print("\nHeld-out test comparison:")
print(test_res.to_string(index=False))

# %%
# Predictions and diagnostics for final comparison

pred_test_all = pd.concat(
    test_preds,
    ignore_index=True,
    sort=False,
)

pred_test_all = add_prediction_quantities(
    pred_test_all,
    quantiles=(
        0.3, 0.4, 0.5, 0.6, 0.7,
        0.75, 0.8, 0.9, 0.95,
        0.99, 0.995, 0.999,
    ),
)

pred_test_all.to_csv(
    OUT_TEST_PRED,
    index=False,
)

model_order = [
    m for m in [
        "Simple fit",
        "GLM both",
        "GAM both",
        best_nn_label,
    ]
    if m in pred_test_all["model"].unique()
]

print("\nModel order for plots:")
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
# ============================================================
# 12. Optional: censoring probability check
# ============================================================

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


print("\nCensoring probability check at threshold:", CENSOR_THRESHOLD)

for model in model_order:
    d = pred_test_all[pred_test_all["model"] == model]

    Fc = egpd_cdf(
        CENSOR_THRESHOLD,
        d["sigma"].to_numpy(),
        d["kappa"].to_numpy(),
        d["xi"].to_numpy(),
    )

    print(
        model,
        "| median F(c) =", np.median(Fc),
        "| mean F(c) =", np.mean(Fc),
    )