#%%
import os
from pathlib import Path
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from downscaling.analysis import (
    prepare_analysis_dataframe,
    save_summary_tables,
    save_occurrence_definition_comparison,
    save_occurrence_latex_table,
    fig_occurrence_contingency,
    fig_occurrence_rates_by_month_hour,
    fig_positive_rainfall_distribution,
    fig_survival_positive_rainfall,
    fig_tipping_bucket_discretization,
    fig_top_correlations,
    fig_lag_correlations,
    fig_scatter_radar_gauge,
    fig_conditional_distribution_by_radar,
    fig_conditional_quantiles_by_radar,
    fig_temporal_intensity_patterns,
    fig_station_spatial_patterns,
    fig_example_events,
)

from downscaling.paths import DOWNSCALING_TABLE, TAB_DIR
from downscaling.paths import make_output_dirs

make_output_dirs()

#%%
# %%
df_raw = pd.read_csv(DOWNSCALING_TABLE, sep=";")
df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True)

print("Raw shape:", df_raw.shape)
print("Missing values in Y_obs:", df_raw["Y_obs"].isna().sum())
print("Proportion of Y_obs = 0:", (df_raw["Y_obs"] == 0).mean())

#%%
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
