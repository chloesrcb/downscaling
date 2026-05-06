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
    compare_train_test_distribution,
    pit_by_observed_bins,
    plot_exponential_qq,
    plot_exponential_qq_single_model,
    plot_parameter_distributions,
    plot_pit_histograms,
    plot_predicted_vs_observed,
    plot_quantile_calibration,
    plot_tail_exceedance_calibration,
    plot_train_test_distribution_shift,
    repeated_test_distribution_checks,
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
from downscaling.tuning import rerank_top_nn_configs, tune_nn_on_outer_train
from downscaling.utils import safe_model_name

make_output_dirs()

#%%
# check initial parameter values
print("init:", {"sigma": SIGMA_INIT, "kappa": KAPPA_INIT, "xi": XI_INIT})

#%%
# Load raw data and prepare modeling dataframe.
df_raw = pd.read_csv(DOWNSCALING_TABLE, sep=";")
df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True)

print("Missing values in Y_obs:", df_raw["Y_obs"].isna().sum())
print("Proportion of Y_obs = 0:", (df_raw["Y_obs"] == 0).mean())


df_model, x_cols27, x_cols_dt0h, x_cols_all = prepare_modeling_dataframe(df_raw)

print("x_cols27 len:", len(x_cols27))
print("x_cols_dt0h len:", len(x_cols_dt0h))
print("x_cols_all len:", len(x_cols_all))
print("df_model shape:", df_model.shape)
print(df_model[["time", "station", "Y_obs"]].head())


#%%
single_cov_col = SINGLE_COV_COL


if single_cov_col not in df_model.columns:
    raise ValueError(f"{single_cov_col} not found in df_model columns.")

print(f"Using single covariate: {single_cov_col}")

df_train_valid, df_test, test_split_info = make_train_valid_test_split(
    df=df_model,
    test_frac=0.10,
    block="30D",
    seed=2026,
)


#%%
# Train/valid/test split summary
for k, v in test_split_info.items():
    if k not in {"test_blocks", "train_valid_blocks"}:
        print(f"{k}: {v}")

print("\nTrain/valid period:")
print(df_train_valid["time"].min(), "->", df_train_valid["time"].max())
print("\nTest period:")
print(df_test["time"].min(), "->", df_test["time"].max())


cv_splits = make_blocked_cv_splits(
    df=df_train_valid,
    n_splits=3,
    block="30D",
    seed=1,
)


for sp in cv_splits:
    print(
        f"fold={sp['fold']} | "
        f"n_train={len(sp['train_idx'])} | "
        f"n_valid={len(sp['valid_idx'])} | "
        f"n_valid_blocks={len(sp['valid_blocks'])}"
    )

# plot the splits to check them
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(
    df_model["time"],
    df_model["Y_obs"],
    alpha=0.18,
    label="Y_obs all",
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
    label="Test",
)
ax.set_xlabel("Time")
ax.set_ylabel("Y_obs")
ax.set_title("")
ax.grid(True)
ax.legend(ncol=2)
plt.tight_layout()
plt.show()

# save the split figure
filename = os.path.join(IM_FOLDER, "cv_splits_with_heldout_test.png")
fig.savefig(filename, dpi=300, bbox_inches="tight")

#%%
# Check covariate distributions in train vs test
x_sets = make_covariate_sets(
    x_cols27=x_cols27,
    x_cols_dt0h=x_cols_dt0h,
    x_cols_all=x_cols_all,
)
print("\nAvailable NN covariate sets:")
for name, cols in x_sets.items():
    print(f"{name:20s}: {len(cols)} variables")


#%%
print("TUNING_PRESET:", TUNING_PRESET)

if TUNING_PRESET == "small":
    nn_param_grid = {
        "variant": ["both"],
        "x_set_name": [
            "central_only",
            "summary_dt0h",
            "all_pixels_dt0h",
            "radar_summaries",
        ],
        "widths": [
            (2,),
            (4,),
            (4, 2),
            (6, 3),
        ],
        "lr": [1e-3, 5e-4],
        "weight_decay": [0.0, 1e-4],
        "batch_size": [64, 128],
        "n_ep": [150],
        "sigma_init": [SIGMA_INIT],
        "kappa_init": [KAPPA_INIT],
        "xi_init": [XI_INIT],
        "censor_threshold": [0.22],
        "init_source": ["default", "gam"],
    }

elif TUNING_PRESET == "medium":
    nn_param_grid = {
        "variant": ["both", "sigma_only", "kappa_only"],
        "x_set_name": [
            "central_only",
            "summary_dt0h",
            "local_pixels_dt0h",
            "all_pixels_dt0h",
            "radar_summaries",
            "radar_all",
            "radar_time",
            "radar_time_space",
        ],
        "widths": [
            (2,),
            (4,),
            (4, 2),
            (6, 3),
            (8, 4),
            (8, 4, 2),
            (12, 6),
        ],
        "lr": [1e-3, 5e-4, 1e-4],
        "weight_decay": [0.0, 1e-5, 1e-4],
        "batch_size": [64, 128, 256],
        "n_ep": [200],
        "sigma_init": [0.40, SIGMA_INIT, 0.80],
        "kappa_init": [0.15, KAPPA_INIT, 0.50],
        "xi_init": [0.15, 0.20, XI_INIT],
        "censor_threshold": [0.22, 0.40],
        "init_source": ["default", "gam"],
    }

elif TUNING_PRESET == "large":
    nn_param_grid = {
        "variant": ["both"],
        "x_set_name": [
            "radar_time_space",
        ],
        "widths": [
            (4, 2),
            (6, 3),
            (8, 4),
            (12, 6),
            (16, 8),
            (24, 12),
            (32, 16),
        ],
        "lr": [1e-3, 5e-4],
        "weight_decay": [0.0, 1e-5, 1e-4],
        "batch_size": [64, 128],
        "n_ep": [200],
        "sigma_init": [0.40, 0.57, 0.80, 1.00],
        "kappa_init": [0.10, 0.15, 0.28, 0.40, 0.70],
        "xi_init": [0.10, 0.15, 0.20, 0.24, 0.30, 0.35, 0.40, 0.50],
        "censor_threshold": [0.22],
        "init_source": ["default", "gam"],
    }

else:
    raise ValueError("TUNING_PRESET must be one of: 'small', 'medium', 'large'.")

#%%
# Run all models on train/valid CV splits, and collect results for comparison.
all_results = []
print("\nRunning stationary model on train_valid CV")
res_m1 = evaluate_stationary_candidate(
    df_model=df_train_valid,
    splits=cv_splits,
    sigma_init=SIGMA_INIT,
    kappa_init=KAPPA_INIT,
    xi_init=XI_INIT,
    censor_threshold=CENSOR_THRESHOLD,
    fix_xi=True,
)
all_results.append(res_m1)

for variant in ["both", "sigma_only", "kappa_only"]:
    print(f"\nRunning GLM 1cov on train_valid CV - variant={variant}")
    res_m2 = evaluate_single_covariate_model(
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
    all_results.append(res_m2)

    print(f"\nRunning GAM 1cov on train_valid CV - variant={variant}")
    res_m3 = evaluate_single_covariate_model(
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
    all_results.append(res_m3)

#%%
# Tuning and evaluating the NN is done in a separate script, since it takes more time.
tuning_df, best_params = tune_nn_on_outer_train(
    df_model=df_train_valid,
    outer_split=cv_splits[0],
    x_sets=x_sets,
    param_grid=nn_param_grid,
    seed=1,
    device=None,
)

print(best_params)
print(tuning_df.head(30).to_string(index=False))
tuning_df.to_csv(OUT_TUNING, index=False)

#%%
# Re-rank top NN configs with validation loss, twCRPS, and sMAD.
rerank_df, best_params = rerank_top_nn_configs(
    df_model=df_train_valid,
    outer_split=cv_splits[0],
    x_sets=x_sets,
    tuning_df=tuning_df,
    top_k=10,
    seed=1,
    device=None,
)

print(best_params)
print(rerank_df.head(20).to_string(index=False))
rerank_df.to_csv(OUT_RERANK, index=False)
pd.DataFrame([best_params]).to_csv(OUT_BEST_PARAMS, index=False)

#%%
# Evaluate the best NN model on all CV folds, and compare to the other models.
best_params_final = best_params.copy()

best_params_final["n_ep"] = 300 # more epochs for final evaluation
res_nn_tuned = evaluate_fixed_nn_model(
    df_model=df_train_valid,
    splits=cv_splits,
    x_sets=x_sets,
    best_params=best_params_final,
    seed=1,
    device=None,
)

#%%
all_results.append(res_nn_tuned)
comparison_res = pd.concat(
    all_results,
    ignore_index=True,
    sort=False,
)
comparison_res.to_csv(
    OUT_COMPARISON,
    index=False,
)


#%%
# TRAIN_VALID CV COMPARISON SUMMARY
summary = summarize_model_comparison(comparison_res)
summary.to_csv(
    OUT_SUMMARY,
    index=False,
)
print(summary.to_string(index=False))
print(summary.head(15).to_string(index=False))

#%%
# delta to best model summary
cols_delta = [
    "model_family",
    "model_type",
    "variant",
    "covariate",
    "n_folds",

    "valid_loss_mean",
    "valid_loss_sum_mean",
    "twcrps_paper_sum_mean",
    "crps_mean",
    "smad_mean",
    "smad_original_mean",
    "err95_mean",
    "err99_mean",

    "valid_loss_delta",
    "valid_loss_sum_delta",
    "twcrps_paper_sum_delta",
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
print(summary_delta.to_string(index=False))


#%%
# Best NN tuning configs summary
print(
    tuning_df
    .groupby(["x_set_name", "init_source"])["valid_loss"]
    .min()
    .sort_values()
    .to_string()
)

#%%
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
    "valid_loss",
    "train_loss",
    "stopped_epoch",
]
cols_show = [c for c in cols_show if c in tuning_df.columns]

print(
    tuning_df
    .sort_values("valid_loss")
    .head(30)[cols_show]
    .to_string(index=False)
)

#%%
# Test set evaluation for the best models
test_rows = []
test_preds = []
row, pred = fit_predict_stationary_test(
    df_train_valid=df_train_valid,
    df_test=df_test,
)

test_rows.append(row)
test_preds.append(pred)

for variant in ["both", "sigma_only", "kappa_only"]:
    row, pred = fit_predict_regression_test(
        df_train_valid=df_train_valid,
        df_test=df_test,
        model_type="glm",
        variant=variant,
        covariate_col=single_cov_col,
    )
    test_rows.append(row)
    test_preds.append(pred)

    row, pred = fit_predict_regression_test(
        df_train_valid=df_train_valid,
        df_test=df_test,
        model_type="gam",
        variant=variant,
        covariate_col=single_cov_col,
    )
    test_rows.append(row)
    test_preds.append(pred)


row, pred = fit_predict_nn_test(
    df_train_valid=df_train_valid,
    df_test=df_test,
    x_sets=x_sets,
    best_params=best_params_final,
    seed=123,
    device=None,
)

#%%
test_rows.append(row)
test_preds.append(pred)
test_res = pd.DataFrame(test_rows)
test_res = test_res.sort_values(
    [
        "test_loss",
        "twcrps_paper_sum",
        "crps_mean",
        "smad",
    ],
    na_position="last",
).reset_index(drop=True)


test_res.to_csv(OUT_TEST, index=False)
print(test_res.to_string(index=False))

#%%
pred_test_all = pd.concat(
    test_preds,
    ignore_index=True,
    sort=False,
)
pred_test_all = add_prediction_quantities(pred_test_all)
pred_test_all.to_csv(OUT_TEST_PRED, index=False)

nn_model_label = f"NN {best_params_final['x_set_name']}"
model_order = [
    "Simple fit",
    "GLM both",
    "GAM both",
    nn_model_label,
]

#%%
plot_exponential_qq(
    pred_test_all,
    model_order=model_order,
    p_low=0.50,
    p_high=0.995,
)

plot_pit_histograms(
    pred_test_all,
    model_order=model_order,
    n_bins=20,
)

plot_predicted_vs_observed(
    pred_test_all,
    model_order=model_order,
    pred_col="q50_pred",
)


plot_predicted_vs_observed(
    pred_test_all,
    model_order=model_order,
    pred_col="mean_pred",
)


plot_predicted_vs_observed(
    pred_test_all,
    model_order=model_order,
    pred_col="q95_pred",
)


plot_quantile_calibration(
    pred_test_all,
    model_order=model_order,
)


exc_df = plot_tail_exceedance_calibration(
    pred_test_all,
    model_order=model_order,
    thresholds=(0.5, 1.0, 2.0, 5.0, 10.0),
)

#%%
dist_check = compare_train_test_distribution(
    df_train_valid,
    df_test,
    y_col="Y_obs",
)

print(dist_check.to_string(index=False))

#%%
pit_bins = pit_by_observed_bins(pred_test_all)
print(pit_bins.to_string(index=False))

#%%
dist_shift = plot_train_test_distribution_shift(
    df_train_valid,
    df_test,
    y_col="Y_obs",
)

#%%
rep_shift = repeated_test_distribution_checks(
    df_model,
    n_repeats=50,
    test_frac=0.10,
    block="30D",
    base_seed=2026,
)


print(
    rep_shift
    .query("set == 'test'")
    [["mean", "q90", "q95", "q99", "p_gt_0.5", "p_gt_1", "p_gt_5"]]
    .describe()
    .to_string()
)


for model_name in model_order:
    plot_exponential_qq_single_model(
        pred_test_all,
        model_name=model_name,
        save_name=f"qq_exponential_{safe_model_name(model_name)}",
    )

