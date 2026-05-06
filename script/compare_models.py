#%%
import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from downscaling.config import (
    XI_INIT,
    SIGMA_INIT,
    KAPPA_INIT,
    CENSOR_THRESHOLD,
    SINGLE_COV_COL,
)

from downscaling.paths import (
    make_output_dirs,
    OUT_TUNING,
    OUT_RERANK,
    OUT_BEST_PARAMS,
    OUT_COMPARISON,
    OUT_SUMMARY,
    OUT_SUMMARY_DELTA,
    OUT_TEST,
    OUT_TEST_PRED,
)

from downscaling.data import (
    load_raw_downscaling_table,
    prepare_modeling_dataframe,
    make_covariate_sets,
)

from downscaling.splits import (
    make_train_valid_test_split,
    make_blocked_cv_splits,
)

from downscaling.evaluation import (
    evaluate_stationary_candidate,
    evaluate_single_covariate_model,
    fit_predict_stationary_test,
    fit_predict_regression_test,
    fit_predict_nn_test,
)

from downscaling.tuning import (
    tune_nn_on_outer_train,
    rerank_top_nn_configs,
    evaluate_fixed_nn_model,
)

from downscaling.scores import summarize_model_comparison

from downscaling.diagnostics import add_prediction_quantities

from downscaling.plotting import (
    plot_exponential_qq,
    plot_pit_histograms,
    plot_predicted_vs_observed,
    plot_quantile_calibration,
    plot_tail_exceedance_calibration,
    plot_exponential_qq_single_model,
    safe_model_name,
)

