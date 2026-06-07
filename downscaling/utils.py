import ast
import numpy as np
from itertools import product
from downscaling.plotting import pretty_predictor_name

STATION_COL_CANDIDATES = ["site", "station", "station_name", "gauge", "name_Y", "site_Y"]

def parse_widths(x):
    """
    Convert widths to tuple.
    """
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    if isinstance(x, np.ndarray):
        return tuple(x.tolist())
    if isinstance(x, str):
        return tuple(ast.literal_eval(x))
    return tuple(x)

def safe_model_name(name: str) -> str:
    """
    Convert a model name to a safe format for file naming.
    """
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("'", "")
    )


def pretty_covariate_name(name: str) -> str:
    """Convert technical covariate names into readable labels for plots."""
    return pretty_predictor_name(name)


def add_grid(ax, axis: str = "both"):
    ax.grid(True, axis=axis, alpha=0.3)


def find_station_col(df: pd.DataFrame) -> str:
    for col in STATION_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"No station column found. Available columns are:\n{df.columns.tolist()}\n"
        "Please set STATION_COL manually."
    )


def make_param_grid(param_grid: dict) -> list[dict]:
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, vals)) for vals in product(*values)]
