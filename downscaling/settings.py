import os
from pathlib import Path

XI_INIT = 0.26
SIGMA_INIT = 0.58
KAPPA_INIT = 0.27

CENSOR_THRESHOLD = 0.22
BUCKET_RESOLUTION = 0.2153
RAIN_THRESHOLD_POSITIVE = 0.0

TIME_COLS = ["tod_sin", "tod_cos", "doy_sin", "doy_cos", "month_sin", "month_cos"]
SPATIAL_COLS = ["lat_Y", "lon_Y", "lat_X", "lon_X"]

ALLOWED_VARIANTS = {"both", "sigma_only", "kappa_only"}
SINGLE_COV_COL = "X_p01_dt0h"
TUNING_PRESET = "large"

SEED = 2026
DEVICE = None

REFERENCE_MODEL = "Stationary EGPD"
MODEL_ORDER = ["Stationary EGPD", "GLM", "GAM", "NN"]
MODEL_ORDER_NO_REF = ["GLM", "GAM", "NN"]

KAPPA_LIMIT = 2.0
LAMBDA_PROP_KAPPA_GT2 = 0.05
LAMBDA_EXCESS_KAPPA = 0.01

STATION_COL_CANDIDATES = ["site", "station", "station_name", "gauge", "name_Y", "site_Y"]

QUANTILES_FOR_DIAGNOSTICS = (
    0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80,
    0.90, 0.95, 0.99, 0.995, 0.999,
)

DATA_FOLDER = Path(os.environ.get("DATA_FOLDER", "../../phd_extremes/data/"))
IM_FOLDER = Path("../../phd_extremes/thesis/resources/images/downscaling/")

DOWNSCALING_TABLE = DATA_FOLDER / "downscaling" / "downscaling_table_named_2019_2025.csv"

OUT_COMPARISON = DATA_FOLDER / "downscaling" / "model_comparison_all_variants.csv"
OUT_TUNING = DATA_FOLDER / "downscaling" / "nn_tuning_history.csv"
OUT_RERANK = DATA_FOLDER / "downscaling" / "nn_tuning_rerank_top_configs.csv"
OUT_BEST_PARAMS = DATA_FOLDER / "downscaling" / "best_nn_params.csv"
OUT_SUMMARY = DATA_FOLDER / "downscaling" / "model_comparison_summary.csv"
OUT_SUMMARY_DELTA = DATA_FOLDER / "downscaling" / "model_comparison_summary_delta_to_best.csv"

OUT_TEST = DATA_FOLDER / "downscaling" / "model_comparison_test.csv"
OUT_TEST_PRED = DATA_FOLDER / "downscaling" / "model_diagnostic_predictions_test.csv"

DIAG_DIR = IM_FOLDER / "diagnostic_figures"
ANA_DIR = IM_FOLDER / "analysis_figures"
FIG_DIR = ANA_DIR / "figures"
TAB_DIR = ANA_DIR / "tables"

OUT_DIR = IM_FOLDER / "outputs"

def make_output_dirs() -> None:
    IM_FOLDER.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_COMPARISON.parent.mkdir(parents=True, exist_ok=True)
