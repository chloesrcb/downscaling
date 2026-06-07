# %%
import os
from pathlib import Path
import sys
from itertools import product
import inspect

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downscaling.config import (
    KAPPA_INIT,
    SIGMA_INIT,
    SINGLE_COV_COL,
    XI_INIT,
)
from downscaling.data import (
    prepare_modeling_dataframe,
    make_covariate_sets,
)
from downscaling.evaluation import evaluate_single_covariate_model
from downscaling.prediction import fit_predict_regression_test
from downscaling.diagnostics import (
    add_prediction_quantities,
    plot_exponential_qq,
    plot_pit_histograms,
    plot_predicted_vs_observed,
    plot_quantile_calibration,
    plot_tail_exceedance_calibration,
    summarize_model_comparison,
)
from downscaling.paths import (
    DOWNSCALING_TABLE,
    IM_FOLDER,
    make_output_dirs,
)
from downscaling.plotting import RESPONSE_LABEL, configure_plot_style, save_png
from downscaling.splits import (
    make_blocked_cv_splits,
    make_train_valid_test_split,
)

make_output_dirs()
configure_plot_style()

# %%
# Output files
OUT_BASELINES_CV = os.path.join(IM_FOLDER, "baselines_cv_results.csv")
OUT_BASELINES_SUMMARY = os.path.join(IM_FOLDER, "baselines_summary.csv")
OUT_BASELINES_TEST = os.path.join(IM_FOLDER, "baselines_test_results.csv")
OUT_BASELINES_TEST_PRED = os.path.join(IM_FOLDER, "baselines_test_predictions.csv")

# %%
# Load data
df_raw = pd.read_csv(DOWNSCALING_TABLE, sep=";")
df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True)

print("Y_obs zero proportion:", (df_raw["Y_obs"] == 0).mean())
print("Y_obs missing:", df_raw["Y_obs"].isna().sum())

df_model, x_cols27, x_cols_dt0h, x_cols_all = prepare_modeling_dataframe(df_raw)

print("df_model shape:", df_model.shape)

# %%
# Splits
df_train_valid, df_test, test_split_info = make_train_valid_test_split(
    df=df_model,
    test_frac=0.10,
    block="15D",
    seed=2026,
)

cv_splits = make_blocked_cv_splits(
    df=df_train_valid,
    n_splits=3,
    block="15D",
    seed=1,
)

print("Train/valid:", df_train_valid["time"].min(), "->", df_train_valid["time"].max())
print("Test:", df_test["time"].min(), "->", df_test["time"].max())

# %%
# Covariates to test
x_sets = make_covariate_sets(
    x_cols27=x_cols27,
    x_cols_dt0h=x_cols_dt0h,
    x_cols_all=x_cols_all,
)

candidate_covariates = [
    "X_p01_dt0h",
    "radar_mean_dt0h",
    "radar_max_dt0h",
]

candidate_covariates = [
    c for c in candidate_covariates
    if c in df_train_valid.columns
]

print("Candidate covariates:")
for c in candidate_covariates:
    print("-", c)

# %%
# Grid for GLM/GAM only
baseline_grid = {
    "model_type": ["glm", "gam"],
    "variant": ["both", "sigma_only", "kappa_only"],
    "covariate_col": candidate_covariates,
    "censor_threshold": [0.22, 0.40, 0.44],
    "xi_init": [XI_INIT, 0.15, 0.20, 0.25],
    "fix_xi": [False],
}

# %%
# CV evaluation: GLM/GAM only
all_results = []

for model_type, variant, covariate_col, censor_threshold, xi_init in product(
    baseline_grid["model_type"],
    baseline_grid["variant"],
    baseline_grid["covariate_col"],
    baseline_grid["censor_threshold"],
    baseline_grid["xi_init"],
):
    print(
        f"\nRunning {model_type.upper()} | "
        f"variant={variant} | cov={covariate_col} | "
        f"censor={censor_threshold} | xi={xi_init}"
    )

    try:
        res = evaluate_single_covariate_model(
            df_model=df_train_valid,
            splits=cv_splits,
            model_type=model_type,
            variant=variant,
            covariate_col=covariate_col,
            sigma_init=SIGMA_INIT,
            kappa_init=KAPPA_INIT,
            xi_init=xi_init,
            censor_threshold=censor_threshold,
        )

        res["censor_threshold"] = censor_threshold
        res["xi_init"] = xi_init

        all_results.append(res)

    except Exception as e:
        print(f"FAILED: {model_type}, {variant}, {covariate_col}: {e}")

# %%
# Save CV results
comparison_res = pd.concat(all_results, ignore_index=True, sort=False)

summary = summarize_model_comparison(comparison_res)
# Ranking focused on probabilistic and tail calibration
rank_cols = []

if "twcrps_sum_mean" in summary.columns:
    summary["rank_twcrps"] = summary["twcrps_sum_mean"].rank(method="average")
    rank_cols.append("rank_twcrps")

if "crps_mean" in summary.columns:
    summary["rank_crps"] = summary["crps_mean"].rank(method="average")
    rank_cols.append("rank_crps")

if "smad_mean" in summary.columns:
    summary["rank_smad"] = summary["smad_mean"].rank(method="average")
    rank_cols.append("rank_smad")

if "mean_abs_err_mean" in summary.columns:
    summary["rank_mae"] = summary["mean_abs_err_mean"].rank(method="average")
    rank_cols.append("rank_mae")

summary["rank_score"] = (
    3.0 * summary["rank_twcrps"]
    + 1.0 * summary.get("rank_crps", 0)
    + 1.0 * summary.get("rank_smad", 0)
    + 0.5 * summary.get("rank_mae", 0)
)

summary = summary.sort_values(
    [
        "rank_score",
        "twcrps_sum_mean",
        "crps_mean",
        "smad_mean",
        "mean_abs_err_mean",
    ],
    na_position="last",
).reset_index(drop=True)

comparison_res.to_csv(OUT_BASELINES_CV, index=False)
summary.to_csv(OUT_BASELINES_SUMMARY, index=False)

print("\nBest CV models, twCRPS-prioritized:")
cols_show = [
    "model_family",
    "model_type",
    "variant",
    "covariate",
    "censor_threshold",
    "xi_init",
    "crps_mean",
    "twcrps_sum_mean",
    "smad_mean",
    "mean_abs_err_mean",
    "rank_score",
]
cols_show = [c for c in cols_show if c in summary.columns]

print(summary.head(30)[cols_show].to_string(index=False))

# %%
# Best model
best = summary.iloc[0].to_dict()

print("\nBest baseline:")
print(best)

# %%# %%
test_rows = []
test_preds = []


def call_with_supported_kwargs(func, **kwargs):
    """Call func with only kwargs accepted by its signature."""
    sig = inspect.signature(func)
    allowed = set(sig.parameters)
    return func(**{k: v for k, v in kwargs.items() if k in allowed})


# On sélectionne depuis comparison_res, car il garde censor_threshold et xi_init
glm_gam_candidates = comparison_res[
    comparison_res["model_type"].astype(str).str.contains("glm|gam", case=False, na=False)
].copy()

print("Candidates found:", len(glm_gam_candidates))

print(
    glm_gam_candidates[
        [
            "model_family",
            "model_type",
            "variant",
            "covariate",
            "censor_threshold",
            "xi_init",
            "twcrps_sum",
        ]
    ].head()
)

# Ranking directement sur les résultats CV non agrégés
sort_cols = [
    "twcrps_sum",
    "crps_mean",
    "smad",
    "mean_abs_err",
]
sort_cols = [c for c in sort_cols if c in glm_gam_candidates.columns]

group_cols = ["model_type", "variant", "covariate"]
group_cols = [c for c in group_cols if c in glm_gam_candidates.columns]

models_to_test = (
    glm_gam_candidates
    .sort_values(sort_cols, na_position="last")
    .groupby(group_cols, as_index=False, dropna=False)
    .head(1)
    .copy()
    .to_dict("records")
)

unique_models_to_test = []
seen = set()

for params in models_to_test:
    key = (
        params.get("model_type"),
        params.get("variant"),
        params.get("covariate"),
        params.get("censor_threshold"),
        params.get("xi_init"),
    )
    if key not in seen:
        seen.add(key)
        unique_models_to_test.append(params)


print("\nModels selected for test:")
for i, p in enumerate(unique_models_to_test, start=1):
    print(
        f"{i:02d} | type={p.get('model_type')} | "
        f"variant={p.get('variant')} | cov={p.get('covariate')} | "
        f"censor={p.get('censor_threshold')} | xi={p.get('xi_init')} | "
        f"cv_twCRPS={p.get('twcrps_sum')}"
    )

# %%
# Fit on train_valid and predict on test
for params in unique_models_to_test:
    raw_model_type = str(params.get("model_type", "")).lower()

    if "glm" in raw_model_type:
        model_type = "glm"
    elif "gam" in raw_model_type:
        model_type = "gam"
    else:
        print(f"Skipping unknown model_type: {raw_model_type}")
        continue

    variant = params.get("variant")
    covariate_col = params.get("covariate")
    censor_threshold = params.get("censor_threshold")
    xi_init = params.get("xi_init")

    try:
        print(
            f"\nTesting {model_type.upper()} | "
            f"variant={variant} | cov={covariate_col} | "
            f"censor={censor_threshold} | xi={xi_init}"
        )

        row, pred = call_with_supported_kwargs(
            fit_predict_regression_test,
            df_train_valid=df_train_valid,
            df_test=df_test,
            model_type=model_type,
            variant=variant,
            covariate_col=covariate_col,
            sigma_init=SIGMA_INIT,
            kappa_init=KAPPA_INIT,
            xi_init=xi_init,
            censor_threshold=censor_threshold,
            fix_xi=False,
        )

        pred["model_name"] = (
            f"{model_type.upper()} | {variant} | {covariate_col} | "
            f"censor={censor_threshold} | xi_init={xi_init}"
        )

        row["model_type"] = model_type
        row["variant"] = variant
        row["covariate"] = covariate_col
        row["censor_threshold"] = censor_threshold
        row["xi_init"] = xi_init
        row["cv_rank_tail_score"] = params.get("rank_tail_score")
        row["cv_twcrps_sum"] = params.get("twcrps_sum")
        row["cv_valid_loss"] = params.get("valid_loss")
        row["cv_err95"] = params.get("err95")
        row["cv_err99"] = params.get("err99")

        test_rows.append(row)
        test_preds.append(pred)

    except Exception as e:
        print(
            f"FAILED test | type={model_type} | "
            f"variant={variant} | cov={covariate_col}: {e}"
        )

# %%
print("Signature fit_predict_regression_test:")
print(inspect.signature(fit_predict_regression_test))
#%%
# Test results
test_res = pd.DataFrame(test_rows)

sort_test_cols = [
    "twcrps_sum",
    "err99",
    "err95",
    "test_loss",
    "crps_mean",
    "smad",
]
sort_test_cols = [c for c in sort_test_cols if c in test_res.columns]

test_res = (
    test_res
    .sort_values(sort_test_cols, na_position="last")
    .reset_index(drop=True)
)

test_res.to_csv(OUT_BASELINES_TEST, index=False)

print("\nTest results:")
print(test_res.to_string(index=False))

# %%
# Test predictions + diagnostic quantities

pred_test_all = pd.concat(test_preds, ignore_index=True, sort=False)
pred_test_all["split"] = "test"

pred_test_all = add_prediction_quantities(pred_test_all)

pred_test_all.to_csv(OUT_BASELINES_TEST_PRED, index=False)

model_order = (
    test_res["model_name"].dropna().tolist()
    if "model_name" in test_res.columns
    else None
)

# %%
# QQ plots on test for all tested models

plot_exponential_qq(
    pred_test_all,
    model_order=model_order,
    p_low=0.50,
    p_high=0.9995,
    save_name="exponential_qq_test.png",
)

# %%
# Other diagnostics on test
plot_quantile_calibration(
    pred_test_all,
    model_order=model_order,
    save_name="quantile_calibration_test.png",
)

plot_tail_exceedance_calibration(
    pred_test_all,
    model_order=model_order,
    thresholds=(1.5, 2.0, 5.0, 8.0, 10.0),
    save_name="tail_exceedance_calibration_test.png",
)

#%%

# table with scores and ranks
test_summary = summarize_model_comparison(test_res) 
test_summary.to_csv(OUT_BASELINES_TEST.replace(".csv", "_summary.csv"), index=False)

print("\nTest summary:")
print(test_summary.to_string(index=False))

#%%
# print scores
print("\nTest scores:")
score_cols = [
    "model_name",
    "twcrps_sum",
    "crps_mean",
    "smad",
    "mean_abs_err_mean",
]
score_cols = [c for c in score_cols if c in test_res.columns]
print(test_res[score_cols].to_string(index=False))

#%%
#%%
# Prediction vs observed on test

plot_predicted_vs_observed(
    pred_test_all,
    model_order=model_order,
    save_name="predicted_vs_observed_test.png",
)

#%%
#%%
#%%
model_name = (
    f"{model_type.upper()} | {variant} | {covariate_col} | "
    f"censor={censor_threshold} | xi_init={xi_init}"
)

pred["model_name"] = model_name

row["model_name"] = model_name
row["model_type"] = model_type
row["variant"] = variant
row["covariate"] = covariate_col
row["censor_threshold"] = censor_threshold
row["xi_init"] = xi_init

#%%
test_res["model_name"] = (
    test_res.apply(
        lambda r: f"{r['model_type'].upper()} | {r['variant']} | {r['covariate']} | "
                  f"censor={r['censor_threshold']} | xi_init={r['xi_init']}",
        axis=1
    )
)
#%%
# Time series observed and predictions in two separate plots

best_model_name = test_res.iloc[0]["model_name"]

df_plot = pred_test_all[pred_test_all["model_name"] == best_model_name].copy()
df_plot = df_plot.sort_values("time")

# Optionnel : choisir une station
if "station" in df_plot.columns:
    station_to_plot = df_plot["station"].value_counts().index[5]
    df_plot = df_plot[df_plot["station"] == station_to_plot].copy()
else:
    station_to_plot = None

title_suffix = f" - station {station_to_plot}" if station_to_plot is not None else ""

# --------
# 1) Observed only
# --------
fig, ax = plt.subplots(figsize=(14, 4))
ax.scatter(df_plot["time"], df_plot["Y_obs"], label=RESPONSE_LABEL, alpha=0.8)
ax.set_xlabel("Time")
ax.set_ylabel(RESPONSE_LABEL)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
save_png(fig, os.path.join(IM_FOLDER, "timeseries_observed_test.png"))
plt.show()

# --------
# 2) Predictions only
# --------
fig, ax = plt.subplots(figsize=(14, 4))
ax.scatter(df_plot["time"], df_plot["mean_pred"], s=8, alpha=0.7, label=r"Predicted mean of $X_{\mathbf{s},t}$")
ax.set_xlabel("Time")
ax.set_ylabel(r"Predicted $X_{\mathbf{s},t}$")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
save_png(fig, os.path.join(IM_FOLDER, "timeseries_predictions_test.png"))
plt.show()

#%%
# on the same plot
fig, ax = plt.subplots(figsize=(14, 4))
ax.scatter(df_plot["time"], df_plot["Y_obs"], label=RESPONSE_LABEL, alpha=0.8)
# ax.scatter(df_plot["time"], df_plot["mean_pred"], alpha=0.7, label=r"Predicted mean of $X_{\mathbf{s},t}$")
ax.scatter(df_plot["time"], df_plot["q95_pred"], alpha=0.7, label=r"Predicted 95% quantile of $X_{\mathbf{s},t}$")
ax.set_xlabel("Time")
ax.set_ylabel(RESPONSE_LABEL)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
save_png(fig, os.path.join(IM_FOLDER, "timeseries_observed_predicted_test.png"))
plt.show()


#%%
fig, ax = plt.subplots(figsize=(12, 4))
ax.scatter(df_plot["time"], df_plot["sigma"])
save_png(fig, os.path.join(IM_FOLDER, "timeseries_sigma_test.png"))
plt.show()

fig, ax = plt.subplots(figsize=(12, 4))
ax.scatter(df_plot["time"], df_plot["kappa"])
save_png(fig, os.path.join(IM_FOLDER, "timeseries_kappa_test.png"))
plt.show()

#%%
# on the same plot
fig, ax = plt.subplots(figsize=(14, 4))
ax.scatter(df_plot["time"], df_plot["Y_obs"], label=RESPONSE_LABEL, alpha=0.8)
ax.scatter(df_plot["time"], df_plot["q50_pred"], alpha=0.7, label=r"Predicted median of $X_{\mathbf{s},t}$")
ax.scatter(df_plot["time"], df_plot["q75_pred"], alpha=0.7, label=r"Predicted 75% quantile of $X_{\mathbf{s},t}$")
ax.set_xlabel("Time")
ax.set_ylabel(RESPONSE_LABEL)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
save_png(fig, os.path.join(IM_FOLDER, "timeseries_observed_predicted_test.png"))
plt.show()

#%% zoom on a specific period
fig, ax = plt.subplots(figsize=(14, 4))
mask_zoom = (df_plot["time"] >= "2022-08-01") & (df_plot["time"] <= "2022-10-30")
ax.scatter(df_plot.loc[mask_zoom, "time"], df_plot.loc[mask_zoom, "Y_obs"], label=RESPONSE_LABEL, alpha=0.8)
ax.scatter(df_plot.loc[mask_zoom, "time"], df_plot.loc[mask_zoom, "q50_pred"], alpha=0.7, label=r"Predicted median of $X_{\mathbf{s},t}$")
ax.scatter(df_plot.loc[mask_zoom, "time"], df_plot.loc[mask_zoom, "q75_pred"], alpha=0.7, label=r"Predicted 75% quantile of $X_{\mathbf{s},t}$")
ax.set_xlabel("Time")
ax.set_ylabel(RESPONSE_LABEL)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
save_png(fig, os.path.join(IM_FOLDER, "timeseries_observed_predicted_zoom_test.png"))
plt.show()
