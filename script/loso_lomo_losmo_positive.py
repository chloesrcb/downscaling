# %%
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downscaling.settings import SINGLE_COV_COL
from downscaling.data import make_covariate_sets, prepare_modeling_dataframe
from downscaling.settings import DOWNSCALING_TABLE, IM_FOLDER, make_output_dirs
from downscaling.plotting import configure_plot_style
from downscaling.utils import find_station_col
from downscaling.tuning import tune_nn_loso
from downscaling.loso import run_final_cv_evaluation

from downscaling.scores import (
    score_loso_by_station,
    summarize_loso_scores,
    add_skill_scores_vs_reference,
    score_one_prediction_table,
)

from downscaling.plotting import (
    MODEL_ORDER,
    MODEL_ORDER_NO_REF,
    REFERENCE_MODEL,
    compact_model_name,
    plot_global_score_and_skill_summary,
    plot_skill_score_boxplots,
    plot_skill_score_heatmap,
    plot_loso_score_boxplot,
    plot_exponential_qq_all_models_zoom,
    plot_exponential_qq_all_models_by_site,
    plot_survival_observed_vs_all_models,
    plot_parameter_boxplots_combined,
)

from downscaling.settings import SEED, DEVICE

# %%
# Configuration
make_output_dirs()
configure_plot_style()

OUT_DIR = Path(IM_FOLDER) / "LOSO_LOMO_LOSMO_intensity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Load data
df_raw = pd.read_csv(DOWNSCALING_TABLE, sep=";")
df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True)

# Save removed stations
df_raw[df_raw["station"].isin(["brives", "hydro", "cines"])].to_csv(
    OUT_DIR / "removed_stations.csv",
    index=False,
)

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

# %%
# Add temporal CV groups
df_model = df_model.copy()
df_model["year"] = df_model["time"].dt.year
df_model["year_month"] = df_model["time"].dt.to_period("M").astype(str)

# LOSMO = leave one station-month cell out
df_model["station_month"] = (
    df_model[STATION_COL].astype(str) + "__" + df_model["year_month"].astype(str)
)

months = sorted(df_model["year_month"].dropna().unique())
station_months = sorted(df_model["station_month"].dropna().unique())

print("Number of months:", len(months))
print("Number of station-month cells:", len(station_months))

# %%
# NN tuning: keep spatial LOSO tuning only
nn_param_grid = {
    "variant": ["both"],
    "x_set_name": ["radar_time_space"],
    "widths": [(8, 4)],
    "lr": [1e-3],
    "weight_decay": [0.0],
    "batch_size": [128],
    "n_ep": [100],
    "sigma_init": [0.53],
    "kappa_init": [0.31],
    "xi_init": [0.18],
    "censor_threshold": [0.3],
    "init_source": ["default"],
    "kappa_max_nn": [1],
    "lambda_kappa": [5],
}

TUNING_STATIONS = ["cnrs", "iem", "poly", "crbm", "um"]

tuning_loso_df, best_params_final = tune_nn_loso(
    df_model=df_model,
    stations_for_tuning=TUNING_STATIONS,
    station_col=STATION_COL,
    x_sets=x_sets,
    param_grid=nn_param_grid,
    seed=SEED,
    device=DEVICE,
)

tuning_loso_df.to_csv(OUT_DIR / "nn_tuning_spatial_loso.csv", index=False)

print("\nBest NN parameters selected by spatial LOSO tuning:")
print(best_params_final)

best_params_final["n_ep"] = 300
pd.DataFrame([best_params_final]).to_csv(
    OUT_DIR / "best_nn_params_selected_by_loso.csv",
    index=False,
)

#%%
# no tuning
best_params_final = {
    "variant": "both",
    "x_set_name": "radar_time_space",
    "widths": (8, 4),
    "lr": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 128,
    "n_ep": 300,
    "sigma_init": 0.53,
    "kappa_init": 0.31,
    "xi_init": 0.18,
    "censor_threshold": 0.3,
    "init_source": "default",
    "kappa_max_nn": 1,
    "lambda_kappa": 5,
}



def compute_global_scores(pred_all: pd.DataFrame, protocol_name: str):
    rows = []

    model_order = [m for m in MODEL_ORDER if m in pred_all["model"].unique()]

    for model in model_order:
        d = pred_all[pred_all["model"] == model].copy()
        if len(d) == 0:
            continue

        scores = score_one_prediction_table(d, alpha=1.0)
        scores["model"] = model
        scores["n_obs"] = len(d)
        scores["cv_protocol"] = protocol_name
        rows.append(scores)

    scores_df = pd.DataFrame(rows)

    scores_skill = add_skill_scores_vs_reference(
        scores_df=scores_df,
        group_cols=[],
        ref_model=REFERENCE_MODEL,
        score_cols=["crps_mean", "crps_sum", "twcrps_sum", "twcrps_mean"],
    )

    return scores_skill


def compute_scores_by_group(pred_all: pd.DataFrame, protocol_name: str):
    rows = []

    for group, dg in pred_all.groupby("left_out_group"):
        for model, dm in dg.groupby("model"):
            scores = score_one_prediction_table(dm, alpha=1.0)
            scores["left_out_group"] = group
            scores["model"] = model
            scores["n"] = len(dm)
            scores["cv_protocol"] = protocol_name
            rows.append(scores)

    scores_df = pd.DataFrame(rows)

    scores_skill = add_skill_scores_vs_reference(
        scores_df=scores_df,
        group_cols=["left_out_group"],
        ref_model=REFERENCE_MODEL,
        score_cols=["crps_mean", "crps_sum", "twcrps_sum", "twcrps_mean"],
    )

    return scores_df, scores_skill


def export_protocol_results(
    pred_all: pd.DataFrame,
    native_rows: pd.DataFrame,
    protocol_name: str,
):
    protocol_dir = OUT_DIR / protocol_name
    protocol_dir.mkdir(parents=True, exist_ok=True)

    pred_all.to_csv(protocol_dir / f"{protocol_name.lower()}_all_predictions.csv", index=False)
    native_rows.to_csv(protocol_dir / f"{protocol_name.lower()}_native_scores.csv", index=False)

    global_scores = compute_global_scores(pred_all, protocol_name)
    global_scores.to_csv(protocol_dir / f"{protocol_name.lower()}_global_skill_scores.csv", index=False)

    scores_by_group, scores_by_group_skill = compute_scores_by_group(pred_all, protocol_name)
    scores_by_group.to_csv(protocol_dir / f"{protocol_name.lower()}_scores_by_group.csv", index=False)
    scores_by_group_skill.to_csv(
        protocol_dir / f"{protocol_name.lower()}_scores_by_group_with_skill.csv",
        index=False,
    )

    print(f"\nGlobal scores for {protocol_name}:")
    keep_cols = [
        "cv_protocol",
        "model",
        "n_obs",
        "crps_mean",
        "crps_sum",
        "crps_skill",
        "twcrps_sum",
        "twcrps_skill",
        "twcrps_mean",
    ]
    keep_cols = [c for c in keep_cols if c in global_scores.columns]
    print(global_scores[keep_cols].round(4).to_string(index=False))

    return global_scores, scores_by_group, scores_by_group_skill


# %%
native_loso, pred_loso_all = run_final_cv_evaluation(
    df_model=df_model,
    groups=stations,
    group_col=STATION_COL,
    group_name="station",
    x_sets=x_sets,
    best_params_final=best_params_final,
    single_cov_col=single_cov_col,
)

global_loso, scores_loso_by_group, scores_loso_by_group_skill = export_protocol_results(
    pred_all=pred_loso_all,
    native_rows=native_loso,
    protocol_name="LOSO",
)

# %%
df_model["time"] = pd.to_datetime(df_model["time"], utc=True)

df_model["year_month"] = (
    df_model["time"]
    .dt.to_period("M")
    .astype(str)
)

df_model["station_month"] = (
    df_model[STATION_COL].astype(str)
    + "__"
    + df_model["year_month"].astype(str)
)

df_model["year"] = df_model["time"].dt.year.astype(str)

df_model["station_year"] = (
    df_model[STATION_COL].astype(str)
    + "__"
    + df_model["year"].astype(str)
)

#%%
# months = sorted(df_model["year_month"].dropna().unique())

# choisir seulement certains mois pour accélérer
months_all = sorted(df_model["year_month"].dropna().unique())

months = [
    "2020-09",
    "2021-09",
    "2022-09",
    "2023-10",
    "2024-09",
]
months = [m for m in months if m in months_all]

native_lomo, pred_lomo_all = run_final_cv_evaluation(
    df_model=df_model,
    groups=months,
    group_col="year_month",
    group_name="month",
    x_sets=x_sets,
    best_params_final=best_params_final,
    single_cov_col=single_cov_col,
)

global_lomo, scores_lomo_by_group, scores_lomo_by_group_skill = export_protocol_results(
    pred_all=pred_lomo_all,
    native_rows=native_lomo,
    protocol_name="LOMO",
)

#%%

df_model["year"] = df_model["time"].dt.year.astype(str)

years = sorted(df_model["year"].dropna().unique())

native_loyo, pred_loyo_all = run_final_cv_evaluation(
    df_model=df_model,
    groups=years,
    group_col="year",
    group_name="year",
    x_sets=x_sets,
    best_params_final=best_params_final,
    single_cov_col=single_cov_col,
)

#%%
global_loyo, scores_loyo_by_group, scores_loyo_by_group_skill = export_protocol_results(
    pred_all=pred_loyo_all,
    native_rows=native_loyo,
    protocol_name="LOYO",
)

# %%
# 3. LOSMO: leave one station-month out
min_n_test = 30

station_month_counts = (
    df_model.groupby("station_year")
    .size()
    .reset_index(name="n")
)

valid_station_months = (
    station_month_counts
    .query("n >= @min_n_test")["station_year"]
    .sort_values()
    .tolist()
)

# get only 10 groups for testing
valid_station_months = ["archiw__2020", "cnrs__2021", "iem__2022", "poly__2023", "crbm__2024"]

native_losmo, pred_losmo_all = run_final_cv_evaluation(
    df_model=df_model,
    groups=valid_station_months,
    group_col="station_year",
    group_name="station_year",
    x_sets=x_sets,
    best_params_final=best_params_final,
    single_cov_col=single_cov_col,
)

# %%

global_losmo, scores_losmo_by_group, scores_losmo_by_group_skill = export_protocol_results(
    pred_all=pred_losmo_all,
    native_rows=native_losmo,
    protocol_name="LOSMO",
)

# %%
# Combine the three validation protocols
pred_all_protocols = pd.concat(
    [pred_loso_all, pred_lomo_all, pred_losmo_all],
    ignore_index=True,
)

global_all_protocols = pd.concat(
    [global_loso, global_lomo, global_losmo],
    ignore_index=True,
)

pred_all_protocols.to_csv(OUT_DIR / "all_protocols_predictions.csv", index=False)
global_all_protocols.to_csv(OUT_DIR / "all_protocols_global_skill_scores.csv", index=False)

print("\nCombined global scores:")
keep_cols = [
    "cv_protocol",
    "model",
    "n_obs",
    "crps_mean",
    "crps_skill",
    "twcrps_sum",
    "twcrps_skill",
    "twcrps_mean",
]
keep_cols = [c for c in keep_cols if c in global_all_protocols.columns]
print(global_all_protocols[keep_cols].round(4).to_string(index=False))

# %%
# Thesis summary table
thesis_protocol_table = global_all_protocols[
    [
        "cv_protocol",
        "model",
        "n_obs",
        "crps_mean",
        "crps_skill",
        "twcrps_sum",
        "twcrps_skill",
        "twcrps_mean",
    ]
].copy()

thesis_protocol_table["model"] = thesis_protocol_table["model"].map(compact_model_name)

thesis_protocol_table.to_csv(
    OUT_DIR / "thesis_table_validation_protocols.csv",
    index=False,
)

latex_table = thesis_protocol_table.to_latex(
    index=False,
    float_format="%.5f",
    caption="Predictive performance under spatial, temporal and spatio-temporal cross-validation protocols.",
    label="tab:validation_protocols_downscaling",
)

print(latex_table)

# %%
# Plots for LOSO only, because it remains the main validation
plot_global_score_and_skill_summary(
    global_loso,
    filename="loso_global_score_and_skill_summary.png",
    out_dir=OUT_DIR / "LOSO",
)

plot_skill_score_boxplots(
    scores_loso_by_group_skill,
    skill_cols=["crps_skill"],
    filename="loso_crps_skill_boxplots.png",
    out_dir=OUT_DIR / "LOSO",
)

plot_skill_score_boxplots(
    scores_loso_by_group_skill,
    skill_cols=["twcrps_skill"],
    filename="loso_twcrps_skill_boxplots.png",
    out_dir=OUT_DIR / "LOSO",
)

plot_skill_score_heatmap(
    scores_loso_by_group_skill.rename(columns={"left_out_group": "left_out_station"}),
    skill_col="crps_skill",
    filename="loso_crps_skill_heatmap_by_site.png",
    out_dir=OUT_DIR / "LOSO",
)

plot_skill_score_heatmap(
    scores_loso_by_group_skill.rename(columns={"left_out_group": "left_out_station"}),
    skill_col="twcrps_skill",
    filename="loso_twcrps_skill_heatmap_by_site.png",
    out_dir=OUT_DIR / "LOSO",
)

# %%
# Diagnostics for LOSO only
plot_exponential_qq_all_models_zoom(
    pred_loso_all,
    pmin_zoom=0.90,
    out_dir=OUT_DIR / "LOSO",
)

for score_col in [
    "twcrps_sum",
    "twcrps_mean",
    "crps_sum",
    "crps_mean",
    "pit_cvm",
    "smad",
    "kappa_q99",
    "prop_kappa_gt_2",
]:
    plot_loso_score_boxplot(
        scores_loso_by_group.rename(columns={"left_out_group": "left_out_station"}),
        score_col,
        out_dir=OUT_DIR / "LOSO",
    )

for site in sorted(pred_loso_all["left_out_station_true"].dropna().unique()):
    plot_exponential_qq_all_models_by_site(
        pred_loso_all.rename(columns={"left_out_station_true": "left_out_station"}),
        site,
        pmin_zoom=0.90,
        out_dir=OUT_DIR / "LOSO",
    )

plot_survival_observed_vs_all_models(
    pred_loso_all,
    models=["NN", "GAM", "GLM", "Stationary EGPD"],
    n_sim_per_obs=100,
    seed=123,
    out_dir=OUT_DIR / "LOSO",
)

plot_parameter_boxplots_combined(
    pred_loso_all,
    models=["NN", "GAM", "GLM"],
    out_dir=OUT_DIR / "LOSO",
)


#%%
def plot_protocol_diagnostics(
    pred_all,
    scores_by_group,
    scores_by_group_skill,
    protocol_name,
):
    protocol_dir = OUT_DIR / protocol_name
    protocol_dir.mkdir(parents=True, exist_ok=True)

    # Pour que les anciennes fonctions LOSO marchent
    scores_plot = scores_by_group.rename(
        columns={"left_out_group": "left_out_station"}
    )
    scores_skill_plot = scores_by_group_skill.rename(
        columns={"left_out_group": "left_out_station"}
    )
    pred_plot = pred_all.rename(
        columns={"left_out_group": "left_out_station"}
    )

    plot_skill_score_boxplots(
        scores_skill_plot,
        skill_cols=["crps_skill"],
        filename=f"{protocol_name.lower()}_crps_skill_boxplots.png",
        out_dir=protocol_dir,
    )

    plot_skill_score_boxplots(
        scores_skill_plot,
        skill_cols=["twcrps_skill"],
        filename=f"{protocol_name.lower()}_twcrps_skill_boxplots.png",
        out_dir=protocol_dir,
    )

    for score_col in [
        "twcrps_sum",
        "twcrps_mean",
        "crps_sum",
        "crps_mean",
        "pit_cvm",
        "smad",
        "kappa_q99",
        "prop_kappa_gt_2",
    ]:
        if score_col in scores_plot.columns:
            plot_loso_score_boxplot(
                scores_plot,
                score_col,
                out_dir=protocol_dir,
            )

    plot_exponential_qq_all_models_zoom(
        pred_plot,
        pmin_zoom=0.90,
        out_dir=protocol_dir,
    )

    plot_survival_observed_vs_all_models(
        pred_plot,
        models=["NN", "GAM", "GLM", "Stationary EGPD"],
        n_sim_per_obs=100,
        seed=123,
        out_dir=protocol_dir,
    )

    plot_parameter_boxplots_combined(
        pred_plot,
        models=["NN", "GAM", "GLM"],
        out_dir=protocol_dir,
    )

#%%
plot_protocol_diagnostics(
    pred_all=pred_loso_all,
    scores_by_group=scores_loso_by_group,
    scores_by_group_skill=scores_loso_by_group_skill,
    protocol_name="LOSO",
)

plot_protocol_diagnostics(
    pred_all=pred_lomo_all,
    scores_by_group=scores_lomo_by_group,
    scores_by_group_skill=scores_lomo_by_group_skill,
    protocol_name="LOMO",
)

plot_protocol_diagnostics(
    pred_all=pred_losmo_all,
    scores_by_group=scores_losmo_by_group,
    scores_by_group_skill=scores_losmo_by_group_skill,
    protocol_name="LOSMO",
)

#%%
protocol_model_table = global_all_protocols[
    ["cv_protocol", "model", "crps_mean", "crps_skill", "twcrps_sum", "twcrps_skill"]
].copy()

protocol_model_table.to_csv(
    OUT_DIR / "protocol_model_comparison.csv",
    index=False,
)

print(protocol_model_table.round(4).to_string(index=False))