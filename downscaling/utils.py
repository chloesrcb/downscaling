import re
import ast
import numpy as np


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
    m = re.match(r"^X_p(\d{2})_dt(-1h|0h|\+1h)$", name)
    if m is not None:
        pixel = int(m.group(1))
        lag = m.group(2)

        if lag == "0h":
            lag_label = "t"
        elif lag == "-1h":
            lag_label = "t - 1 h"
        elif lag == "+1h":
            lag_label = "t + 1 h"
        else:
            lag_label = lag

        return f"Radar grid cell {pixel}, {lag_label}"

    replacements = {
        "radar_max": "Radar maximum",
        "radar_mean": "Radar mean",
        "radar_sum": "Radar sum",
        "tod_sin": "Time of day, sine",
        "tod_cos": "Time of day, cosine",
        "doy_sin": "Day of year, sine",
        "doy_cos": "Day of year, cosine",
        "month_sin": "Month, sine",
        "month_cos": "Month, cosine",
        "lat_Y": "Gauge latitude",
        "lon_Y": "Gauge longitude",
        "lat_X": "Radar cell latitude",
        "lon_X": "Radar cell longitude",
        "Y_obs": r"Gauge rainfall $Y_{obs}$",
    }

    return replacements.get(name, name)


def add_grid(ax, axis: str = "both"):
    ax.grid(True, axis=axis, alpha=0.3)