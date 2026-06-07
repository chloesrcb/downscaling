# %%
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downscaling.config import SINGLE_COV_COL
from downscaling.data import make_covariate_sets, prepare_modeling_dataframe
from downscaling.paths import DOWNSCALING_TABLE, IM_FOLDER, make_output_dirs
from downscaling.plotting import configure_plot_style

from downscaling.utils import find_station_col
from downscaling.tuning import tune_nn_loso
from downscaling.loso import run_final_loso_evaluation
from downscaling.plotting import (
    plot_global_score_and_skill_summary,
    plot_skill_score_boxplots,
    plot_skill_score_heatmap,
    plot_exponential_qq_all_models,
    plot_exponential_qq_all_models_by_site,
    plot_exponential_qq_all_models_by_radar_tercile,
    plot_observed_vs_simulated_density,
    plot_survival_observed_vs_simulated,
    plot_pit_histogram,
    plot_loso_score_boxplot
)
from downscaling.scores import (
    score_by_radar_tercile,
    score_loso_by_station,
    summarize_loso_scores,
    add_skill_scores_vs_reference,
    score_one_prediction_table,
)

from downscaling.plotting import MODEL_ORDER, MODEL_ORDER_NO_REF, REFERENCE_MODEL, compact_model_name, model_color, savefig
from downscaling.config import SEED, DEVICE, KAPPA_INIT, SIGMA_INIT, XI_INIT, LAMBDA_PROP_KAPPA_GT2, LAMBDA_EXCESS_KAPPA

# %%
# ============================================================
# Configuration
# ============================================================
make_output_dirs()
configure_plot_style()

OUT_DIR = Path(IM_FOLDER) / "leave_one_site_out_tuning_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# %%
# Load data
df_raw = pd.read_csv(DOWNSCALING_TABLE, sep=";")
df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True)

df_model, x_cols27, x_cols_dt0h, x_cols_all = prepare_modeling_dataframe(df_raw)

STATION_COL = find_station_col(df_model)
print("Using station column:", STATION_COL)

single_cov_col = SINGLE_COV_COL
if single_cov_col not in df_model.columns:
    raise ValueError(f"{single_cov_col} not found in df_model.")

x_sets = make_covariate_sets(
    x_cols27=x_cols27,
    x_cols_dt0h=x_cols_dt0h,
    x_cols_all=x_cols_all,
)

stations = sorted(df_model[STATION_COL].dropna().unique())
print("Number of stations:", len(stations))
print(stations)

# remove brives hydro and cines sites
df_raw[df_raw["station"].isin(["brives", "hydro", "cines"])].to_csv(OUT_DIR / "removed_stations.csv", index=False)


# %%
# NN LOSO tuning
nn_param_grid = {
    "variant": ["both"],
    "x_set_name": ["radar_time_space"],
    "widths": [(8, 4)],
    "lr": [1e-3],
    "weight_decay": [0.0],
    "batch_size": [128],
    "n_ep": [100],
    "sigma_init": [0.5, SIGMA_INIT, 0.6],
    "kappa_init": [0.2, KAPPA_INIT, 0.3],
    "xi_init": [0.15,  0.20, 0.26, 0.3],
    "censor_threshold": [0.22, 0.3, 0.44],
    "init_source": ["default"],
    "kappa_max_nn": [1],
    "lambda_kappa": [5],
}

TUNING_STATIONS = ["cnrs", "cefe", "iem", "crbm", "poly"]
# TUNING_STATIONS = select_tuning_stations(df_model, stations, STATION_COL, n_tuning_stations=5)

tuning_loso_df, best_params_final = tune_nn_loso(
    df_model=df_model,
    stations_for_tuning=TUNING_STATIONS,
    station_col=STATION_COL,
    x_sets=x_sets,
    param_grid=nn_param_grid,
    seed=SEED,
    device=DEVICE,
)

tuning_loso_df.to_csv(OUT_DIR / "loso_nn_tuning_spatial.csv", index=False)

print("\nBest NN parameters selected by LOSO tuning:")
print(best_params_final)

best_params_final["n_ep"] = 300
pd.DataFrame([best_params_final]).to_csv(OUT_DIR / "loso_best_nn_params.csv", index=False)


# %%
# Final LOSO evaluation
native_rows, pred_loso_all = run_final_loso_evaluation(
    df_model=df_model,
    stations=stations,
    station_col=STATION_COL,
    x_sets=x_sets,
    best_params_final=best_params_final,
    single_cov_col=single_cov_col,
)

native_rows.to_csv(OUT_DIR / "loso_native_scores.csv", index=False)
pred_loso_all.to_csv(OUT_DIR / "loso_all_predictions.csv", index=False)

print("Saved predictions to:", OUT_DIR / "loso_all_predictions.csv")

MODEL_ORDER = [m for m in MODEL_ORDER if m in pred_loso_all["model"].unique()]
MODEL_ORDER_NO_REF = [m for m in MODEL_ORDER_NO_REF if m in pred_loso_all["model"].unique()]


# %%
# Scores, skill scores, and thesis tables
loso_scores_by_station = score_loso_by_station(pred_loso_all)
loso_scores_by_station.to_csv(OUT_DIR / "loso_scores_by_station.csv", index=False)

summary_loso = summarize_loso_scores(loso_scores_by_station)
summary_loso.to_csv(OUT_DIR / "loso_summary_delta_to_best.csv", index=False)

loso_scores_by_station_skill = add_skill_scores_vs_reference(
    scores_df=loso_scores_by_station,
    group_cols=["left_out_station"],
    ref_model=REFERENCE_MODEL,
    score_cols=["crps_mean", "twcrps_sum", "twcrps_mean"],
)
loso_scores_by_station_skill.to_csv(OUT_DIR / "loso_scores_by_station_with_skill_scores.csv", index=False)

global_rows = []
for model in MODEL_ORDER:
    d = pred_loso_all[pred_loso_all["model"] == model].copy()
    if len(d) == 0:
        continue
    scores = score_one_prediction_table(d, alpha=1.0)
    scores["model"] = model
    scores["n_obs"] = len(d)
    global_rows.append(scores)

global_scores = pd.DataFrame(global_rows)
global_scores_skill = add_skill_scores_vs_reference(
    scores_df=global_scores,
    group_cols=[],
    ref_model=REFERENCE_MODEL,
    score_cols=["crps_mean", "twcrps_sum", "twcrps_mean"],
)
global_scores_skill.to_csv(OUT_DIR / "loso_global_skill_scores.csv", index=False)

cols_display = [
    "model", "n_sites", "n_obs",
    "twcrps_sum_total", "twcrps_sum_total_delta", "twcrps_sum_total_rel_delta_pct",
    "twcrps_sum_mean_site", "twcrps_sum_mean_site_delta",
    "twcrps_mean_mean_site", "crps_mean", "crps_mean_delta",
    "smad_mean", "smad_mean_delta", "pit_cvm_mean",
    "kappa_q99_mean", "prop_kappa_gt_2_mean",
]
cols_display = [c for c in cols_display if c in summary_loso.columns]

print("\nLOSO summary with delta to best:")
print(summary_loso[cols_display].round(4).to_string(index=False))

print("\nGlobal LOSO skill scores relative to Stationary EGPD:")
print(
    global_scores_skill[
        ["model", "n_obs", "crps_mean", "crps_skill", "twcrps_sum", "twcrps_skill", "twcrps_mean"]
    ].round(4).to_string(index=False)
)

thesis_delta_table = summary_loso[cols_display].copy()
thesis_delta_table["model"] = thesis_delta_table["model"].map(compact_model_name)
thesis_delta_table.to_csv(OUT_DIR / "loso_thesis_table_delta_to_best.csv", index=False)

thesis_skill_table = global_scores_skill[
    ["model", "crps_mean", "crps_skill", "twcrps_sum", "twcrps_skill", "twcrps_mean"]
].copy()
thesis_skill_table["model"] = thesis_skill_table["model"].map(compact_model_name)
thesis_skill_table.to_csv(OUT_DIR / "loso_thesis_table_skill_scores.csv", index=False)

print(thesis_skill_table.round(4).to_string(index=False))

# %%
plot_global_score_and_skill_summary(
    global_scores_skill,
    filename="loso_global_score_and_skill_summary.png",
)

plot_skill_score_boxplots(
    loso_scores_by_station_skill,
    skill_cols=["crps_skill"],
    filename="loso_crps_skill_boxplots.png",
)

#%%
plot_skill_score_boxplots(
    loso_scores_by_station_skill,
    skill_cols=["twcrps_skill"],
    filename="loso_twcrps_skill_boxplots.png",
)

#%%
plot_skill_score_heatmap(
    loso_scores_by_station_skill,
    skill_col="crps_skill",
    filename="loso_crps_skill_heatmap_by_site.png",
)

#%%
plot_skill_score_heatmap(
    loso_scores_by_station_skill,
    skill_col="twcrps_skill",
    filename="loso_twcrps_skill_heatmap_by_site.png",
)

#%%
plot_exponential_qq_all_models(pred_loso_all)



# %%
# score boxplots and delta-to-best plots
for score_col in ["twcrps_sum", "twcrps_mean", "crps_mean", "pit_cvm", "smad", "kappa_q99", "prop_kappa_gt_2"]:
    plot_loso_score_boxplot(loso_scores_by_station, score_col)



# %%
# diagnostics by site and radar tercile
RUN_SITE_QQ = True
RUN_RADAR_TERCILE_QQ = True
SITE_TO_CHECK = "poly"

pred_loso_all_with_radar = attach_radar_columns_to_predictions(
    pred_df=pred_loso_all,
    df_model=df_model,
    stations=stations,
    station_col=STATION_COL,
    model_order=MODEL_ORDER,
)

if RUN_SITE_QQ:
    for site in sorted(pred_loso_all["left_out_station"].dropna().unique()):
        plot_exponential_qq_all_models_by_site(pred_loso_all, site)

if RUN_RADAR_TERCILE_QQ:
    for site in sorted(pred_loso_all_with_radar["left_out_station"].dropna().unique()):
        plot_exponential_qq_all_models_by_radar_tercile(
            pred_df=pred_loso_all_with_radar,
            site=site,
            use_dt0h_only=True,
            radar_summary="mean",
        )

scores_poly_terciles = score_by_radar_tercile(pred_loso_all_with_radar, site=SITE_TO_CHECK)
scores_poly_terciles.to_csv(OUT_DIR / f"scores_by_radar_tercile_{SITE_TO_CHECK}.csv", index=False)

print("\nScores by radar tercile for", SITE_TO_CHECK)
print(
    scores_poly_terciles[
        ["model", "radar_tercile", "n", "twcrps_sum", "crps_mean", "smad", "kappa_q99", "prop_kappa_gt_2"]
    ].sort_values(["radar_tercile", "twcrps_sum"]).round(4).to_string(index=False)
)


# %%
# PIT / density / survival diagnostics
plot_pit_histogram(pred_loso_all, model)
plot_observed_vs_simulated_density(pred_loso_all, model, n_sim_per_obs=50)
plot_survival_observed_vs_simulated(pred_loso_all, model, n_sim_per_obs=50)


# %%
# Export twCRPS by site/model tables
twcrps_site_model = (
    loso_scores_by_station[["left_out_station", "model", "twcrps_sum", "twcrps_mean", "n"]]
    .copy()
    .sort_values(["left_out_station", "twcrps_sum"])
)
twcrps_site_model.to_csv(OUT_DIR / "twcrps_by_site_and_model.csv", index=False)

print("\ntwCRPS by site and model:")
print(twcrps_site_model.round(4).to_string(index=False))

twcrps_wide = twcrps_site_model.pivot(
    index="left_out_station",
    columns="model",
    values="twcrps_sum",
)
twcrps_wide.to_csv(OUT_DIR / "twcrps_by_site_and_model_wide.csv")

twcrps_delta = twcrps_wide.sub(twcrps_wide.min(axis=1), axis=0)
twcrps_delta.to_csv(OUT_DIR / "twcrps_delta_to_best_by_site.csv")

print("\ntwCRPS delta to best by site:")
print(twcrps_delta.round(2).to_string())


#%%
import matplotlib.pyplot as plt
import pandas as pd
from downscaling.plotting import plot_parameter_boxplots_combined

plot_parameter_boxplots_combined(pred_loso_all, models=["NN", "GAM", "GLM"])