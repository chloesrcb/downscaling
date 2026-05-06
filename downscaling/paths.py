import os
from pathlib import Path

DATA_FOLDER = Path(os.environ.get("DATA_FOLDER", "../../phd_extremes/data/"))

IM_FOLDER = Path("../../phd_extremes/thesis/resources/images/downscaling/")


DOWNSCALING_TABLE = DATA_FOLDER / "downscaling" / "downscaling_table_named_2019_2024.csv"

OUT_COMPARISON = DATA_FOLDER / "downscaling" / "model_comparison_all_variants.csv"
OUT_TUNING = DATA_FOLDER / "downscaling" / "nn_tuning_history.csv"
OUT_RERANK = DATA_FOLDER / "downscaling" / "nn_tuning_rerank_top_configs.csv"
OUT_BEST_PARAMS = DATA_FOLDER / "downscaling" / "best_nn_params.csv"
OUT_SUMMARY = DATA_FOLDER / "downscaling" / "model_comparison_summary.csv"
OUT_SUMMARY_DELTA = DATA_FOLDER / "downscaling" / "model_comparison_summary_delta_to_best.csv"

OUT_TEST = DATA_FOLDER / "downscaling" / "model_comparison_test.csv"
OUT_TEST_PRED = DATA_FOLDER / "downscaling" / "model_diagnostic_predictions_test.csv"

DIAG_DIR = IM_FOLDER / "diagnostic_figures"
OUT_DIR = IM_FOLDER / "analysis_figures"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"


def make_output_dirs() -> None:
    IM_FOLDER.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    OUT_COMPARISON.parent.mkdir(parents=True, exist_ok=True)