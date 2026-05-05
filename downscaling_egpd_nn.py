
# %%
import os
import re
import ast
import math
import inspect
from dataclasses import dataclass
from itertools import product
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from scipy.optimize import minimize
from scipy.interpolate import BSpline
from scipy.special import beta as beta_function

from functions_dwscl import (
    EGPDPINN_NNOnly,
    train_egpd_nn_only,
    predict_params,
)

# %%
# Namespace and paths
DATA_FOLDER = os.environ.get("DATA_FOLDER", "../phd_extremes/data/")
IM_FOLDER = "../phd_extremes/thesis/resources/images/downscaling/"
DOWNSCALING_TABLE = os.path.join(
    DATA_FOLDER,
    "downscaling/downscaling_table_named_2019_2024.csv"
)

OUT_COMPARISON = os.path.join(DATA_FOLDER, "downscaling/model_comparison_all_variants.csv")
OUT_TUNING = os.path.join(DATA_FOLDER, "downscaling/nn_tuning_history.csv")
OUT_RERANK = os.path.join(DATA_FOLDER, "downscaling/nn_tuning_rerank_top_configs.csv")
OUT_BEST_PARAMS = os.path.join(DATA_FOLDER, "downscaling/best_nn_params.csv")
OUT_SUMMARY = os.path.join(DATA_FOLDER, "downscaling/model_comparison_summary.csv")
OUT_SUMMARY_DELTA = os.path.join(DATA_FOLDER, "downscaling/model_comparison_summary_delta_to_best.csv")
OUT_PRED = os.path.join(DATA_FOLDER, "downscaling/model_diagnostic_predictions.csv")

DIAG_DIR = os.path.join(IM_FOLDER, "diagnostic_figures")

os.makedirs(IM_FOLDER, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_COMPARISON), exist_ok=True)

# %%
# Global params
XI_INIT = 0.24
SIGMA_INIT = 0.57
KAPPA_INIT = 0.28

CENSOR_THRESHOLD = 0.22
BUCKET_RESOLUTION = 0.2153

TIME_COLS = ["tod_sin", "tod_cos", "doy_sin", "doy_cos", "month_sin", "month_cos"]
SPATIAL_COLS = ["lat_Y", "lon_Y", "lat_X", "lon_X"]

ALLOWED_VARIANTS = {"both", "sigma_only", "kappa_only"}

print("init:", {"sigma": SIGMA_INIT, "kappa": KAPPA_INIT, "xi": XI_INIT})


# %%
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


def save_diag_fig(fig, name: str, dpi: int = 300):
    """
    Save diagnostic figure in both PNG and PDF formats.
    """
    png_path = os.path.join(DIAG_DIR, f"{name}.png")
    pdf_path = os.path.join(DIAG_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


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


# %%
def make_time_blocks(df: pd.DataFrame, block: str = "30D") -> pd.Series:
    """
    Create time blocks for blocked cross-validation.
    They are randomly selected.
    """
    return df["time"].dt.floor(block)


def make_blocked_cv_splits(df: pd.DataFrame, n_splits: int = 5,
                            block: str = "30D",
                            seed: int = 1,):
    """
    Create blocked cross-validation splits based on time blocks.
    Blocks are randomly selected and assigned to folds.
    """
    rng = np.random.default_rng(seed)

    block_labels = make_time_blocks(df, block=block)
    unique_blocks = np.array(sorted(block_labels.unique()))
    rng.shuffle(unique_blocks)

    folds = np.array_split(unique_blocks, n_splits)

    splits = []
    for k in range(n_splits):
        valid_blocks = set(folds[k])
        train_idx = df.index[~block_labels.isin(valid_blocks)].to_numpy()
        valid_idx = df.index[block_labels.isin(valid_blocks)].to_numpy()

        splits.append({
            "fold": k,
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "valid_blocks": sorted(valid_blocks),
        })

    return splits


def make_single_split_from_train(
                                df_train: pd.DataFrame,
                                train_frac: float = 0.8,
                                block: str = "30D",
                                seed: int = 123,
                            ):
    """
    Make a single train/validation split from the training data, using time blocks.
    """
    d = df_train.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True)

    block_labels = d["time"].dt.floor(block)
    unique_blocks = np.array(sorted(block_labels.unique()))

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_blocks)

    n_train_blocks = int(np.floor(train_frac * len(unique_blocks)))
    n_train_blocks = max(1, min(n_train_blocks, len(unique_blocks) - 1))

    inner_train_blocks = set(unique_blocks[:n_train_blocks])

    inner_train_idx = d.index[block_labels.isin(inner_train_blocks)].to_numpy()
    inner_valid_idx = d.index[~block_labels.isin(inner_train_blocks)].to_numpy()

    return {
        "fold": 0,
        "train_idx": inner_train_idx,
        "valid_idx": inner_valid_idx,
        "valid_blocks": sorted(set(unique_blocks[n_train_blocks:])),
    }


# %%

def get_x_cols27_downscaling(df: pd.DataFrame) -> list[str]:
    """
    Get the cube radar ie the 27 columns named X_pXX_dt... in the correct order.
    'p01' is the central pixel.
    The order is by increasing distance (p01, p02, ..., p09, p10, ..., p27) and 
    then by time (dt-1h, dt0h, dt+1h)."""
    pat = re.compile(r"^X_p(\d{2})_dt(-1h|0h|\+1h)$")
    cols = [c for c in df.columns if pat.match(c)]
    order_dt = {"-1h": 0, "0h": 1, "+1h": 2}

    def key(c):
        m = pat.match(c)
        return (int(m.group(1)), order_dt[m.group(2)])

    return sorted(cols, key=key)


def qeGPD(prob, sigma, kappa, xi, eps=1e-12, kappa_min=5e-2, xi_min=1e-6):
    """
    Quantile function of the EGPD distribution.
    """
    prob = np.asarray(prob, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), eps)
    kappa = np.maximum(np.asarray(kappa, dtype=float), kappa_min)
    xi = np.maximum(np.asarray(xi, dtype=float), xi_min)

    prob = np.clip(prob, eps, 1.0 - eps)
    p1k = np.exp(np.log(prob) / kappa)
    t = np.maximum(1.0 - p1k, eps)

    return sigma / xi * (np.power(t, -xi) - 1.0)


def egpd_cdf(y, sigma, kappa, xi, eps=1e-12, kappa_min=5e-2, xi_min=1e-6):
    """
    CDF of the EGPD distribution.
    F(y) = G(y)^kappa where G is the GPD CDF.
    """
    y = np.asarray(y, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), eps)
    kappa = np.maximum(np.asarray(kappa, dtype=float), kappa_min)
    xi = np.maximum(np.asarray(xi, dtype=float), xi_min)

    t = 1.0 + xi * y / sigma
    t = np.maximum(t, eps)

    a = np.power(t, -1.0 / xi)
    inner = np.clip(1.0 - a, eps, 1.0 - eps)

    return np.clip(np.power(inner, kappa), eps, 1.0 - eps)


def egpd_logpdf(y, sigma, kappa, xi, eps=1e-12):
    """
    Log-PDF of the EGPD distribution.
    """
    y = np.asarray(y, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), eps)
    kappa = np.maximum(np.asarray(kappa, dtype=float), eps)
    xi = np.maximum(np.asarray(xi, dtype=float), eps)

    z = np.maximum(1.0 + xi * y / sigma, eps)
    t = 1.0 - np.power(z, -1.0 / xi)
    t = np.clip(t, eps, 1.0 - eps)

    logf = (
        np.log(kappa)
        - np.log(sigma)
        - (1.0 / xi + 1.0) * np.log(z)
        + (kappa - 1.0) * np.log(t)
    )

    return logf


def egpd_left_censored_loglik(y, sigma, kappa, xi, c=0.22, eps=1e-12):
    """Log-likelihood of the left-censored EGPD distribution.
    For y <= c, contribution is log(F(c))."""
    y = np.asarray(y, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    kappa = np.asarray(kappa, dtype=float).reshape(-1)
    xi = np.asarray(xi, dtype=float).reshape(-1)

    unc = y > c # uncensored
    cen = ~unc # left-censored

    ll = np.zeros_like(y, dtype=float)

    # Uncensored contribution with log PDF
    if np.any(unc):
        ll[unc] = egpd_logpdf(
            y[unc],
            sigma[unc],
            kappa[unc],
            xi[unc],
            eps=eps,
        )

    # Censored contribution with log CDF at c
    if np.any(cen):
        Fc = egpd_cdf(
            np.full(np.sum(cen), c),
            sigma[cen],
            kappa[cen],
            xi[cen],
            eps=eps,
        )
        ll[cen] = np.log(np.clip(Fc, eps, 1.0))

    return ll


def egpd_left_censored_nll(y, sigma, kappa, xi, c=0.22, eps=1e-12):
    """
    Negative log-likelihood of the left-censored EGPD distribution, averaged over observations.
    For y <= c, contribution is log(F(c))."""
    ll = egpd_left_censored_loglik(
        y,
        sigma,
        kappa,
        xi,
        c=c,
        eps=eps,
    )
    nll = -np.mean(ll)

    if not np.isfinite(nll):
        return 1e6

    return float(nll)


def egpd_left_censored_nll_sum(y, sigma, kappa, xi, c=0.22, eps=1e-12,):
    """Negative log-likelihood of the left-censored EGPD distribution, summed over observations."""
    ll = egpd_left_censored_loglik(
        y,
        sigma,
        kappa,
        xi,
        c=c,
        eps=eps,
    )
    nll = -np.sum(ll)

    if not np.isfinite(nll):
        return 1e12

    return float(nll)


def compute_pit(y, sigma, kappa, xi):
    """
    Compute the Probability Integral Transform (PIT) values for observations y given EGPD parameters.
    """
    return egpd_cdf(y, sigma, kappa, xi)


def predicted_mean_mc(sigma, kappa, xi, n_mc=1000, seed=123):
    """
    Compute the predicted mean of the EGPD distribution using Monte Carlo sampling.
    """
    rng = np.random.default_rng(seed)

    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    kappa = np.asarray(kappa, dtype=float).reshape(-1)
    xi = np.asarray(xi, dtype=float).reshape(-1)

    u = rng.uniform(size=(n_mc, len(sigma)))

    samples = qeGPD(
        u,
        sigma[None, :],
        kappa[None, :],
        xi[None, :],
    )

    return samples.mean(axis=0)


def egpd_mean(sigma, kappa, xi, eps=1e-12):
    """
    Mean of the EGPD.
    If F(y) = G(y)^kappa with G a GPD CDF, then:
    E[Y] = sigma / xi * (kappa * B(kappa, 1 - xi) - 1).
    """
    sigma = np.asarray(sigma, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    xi = np.asarray(xi, dtype=float)

    out = np.full_like(sigma, np.nan, dtype=float)
    ok = xi < 1.0 # ok for xi < 1 to have finite mean

    out[ok] = (
        sigma[ok] / np.maximum(xi[ok], eps)
        * (
            kappa[ok] * beta_function(kappa[ok], 1.0 - xi[ok])
            - 1.0
        )
    )

    return out


# %%
# Scores: CRPS / twCRPS / sMAD

def make_irregular_thresholds_from_data(y: np.ndarray) -> np.ndarray:
    """
    Make irregular thresholds for CRPS integration based on the data distribution.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    ymax = float(np.max(y))
    upper = max(1.05 * ymax, 10.0)

    parts = [np.linspace(0.01, min(2.0, upper), 80)]

    if upper > 2.0:
        parts.append(np.linspace(2.0, min(10.0, upper), 80))

    if upper > 10.0:
        parts.append(np.linspace(10.0, upper, 80))

    thr = np.unique(np.concatenate(parts))
    thr = thr[(thr > 0) & (thr <= upper)]
    thr.sort()

    return thr


def make_rain_twcrps_thresholds(y: np.ndarray,
                                q_low: float = 0.95,
                                q_high: float = 0.995,
                                n_thresholds: int = 24,
                            ) -> np.ndarray:
    """
    Make thresholds for rain twCRPS based on quantiles of the positive rainfall values.
    """
    y = np.asarray(y, dtype=float)
    y_pos = y[np.isfinite(y) & (y > 0)]

    if len(y_pos) == 0:
        raise ValueError("No positive rainfall values available.")

    probs = np.linspace(q_low, q_high, n_thresholds)
    thresholds = np.unique(np.quantile(y_pos, probs))

    if len(thresholds) < 2:
        raise ValueError("Not enough unique thresholds for twCRPS.")

    return thresholds


def twcrps_weight_rain_power(thresholds: np.ndarray,
                            u0: float,
                            alpha: float = 1.0,
                            normalize: bool = True,
                        ) -> np.ndarray:
    """
    Compute power-based weights for rain twCRPS.
    w(u) = (u / u0)^alpha for u >= u0, and 0 otherwise."""
    thresholds = np.asarray(thresholds, dtype=float)

    w = np.zeros_like(thresholds, dtype=float)
    mask = thresholds >= u0

    if alpha == 0:
        w[mask] = 1.0
    else:
        w[mask] = (thresholds[mask] / u0) ** alpha

    if normalize and np.max(w) > 0:
        w = w / np.max(w)

    return w


def crps_integrated_egpd(y: np.ndarray,
                        sigma: np.ndarray,
                        kappa: np.ndarray,
                        xi: np.ndarray,
                        thresholds: np.ndarray,
                        weights: Optional[np.ndarray] = None,
                    ) -> dict:
    """
    Compute the CRPS of the EGPD distribution by integrating over thresholds.
     CRPS = \int (F(u) - I(y <= u))^2 w(u) du
     where F is the EGPD CDF and w(u) are optional weights.
     """
    y = np.asarray(y, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    kappa = np.asarray(kappa, dtype=float).reshape(-1)
    xi = np.asarray(xi, dtype=float).reshape(-1)
    thresholds = np.asarray(thresholds, dtype=float).reshape(-1)

    if np.any(np.diff(thresholds) <= 0):
        raise ValueError("thresholds must be strictly increasing.")

    if weights is None:
        weights = np.ones_like(thresholds, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != len(thresholds):
            raise ValueError("weights and thresholds must have same length.")

    F_mat = np.column_stack([
        egpd_cdf(u, sigma, kappa, xi) for u in thresholds
    ])

    I_mat = (y[:, None] <= thresholds[None, :]).astype(float)

    integrand = weights[None, :] * (F_mat - I_mat) ** 2
    pointwise = np.trapezoid(integrand, x=thresholds, axis=1)

    return {
        "score_mean": float(np.mean(pointwise)),
        "score_sum": float(np.sum(pointwise)),
        "pointwise_scores": pointwise,
    }


def twcrps_discrete_sum_egpd(y: np.ndarray,
                            sigma: np.ndarray,
                            kappa: np.ndarray,
                            xi: np.ndarray,
                            thresholds: np.ndarray,
                            weights: np.ndarray,
                        ) -> dict:
    """
    Compute the discrete sum version of the twCRPS for EGPD.
     twCRPS = sum_j w(u_j) * (F(u_j) - I(y <= u_j))^2
     where F is the EGPD CDF and w(u_j) are weights for each threshold u_j.
     This is a discrete approximation of the integrated CRPS.
     """
    y = np.asarray(y, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    kappa = np.asarray(kappa, dtype=float).reshape(-1)
    xi = np.asarray(xi, dtype=float).reshape(-1)
    thresholds = np.asarray(thresholds, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    if len(weights) != len(thresholds):
        raise ValueError("weights and thresholds must have same length.")

    F_mat = np.column_stack([
        egpd_cdf(u, sigma, kappa, xi) for u in thresholds
    ])

    I_mat = (y[:, None] <= thresholds[None, :]).astype(float)

    score_mat = weights[None, :] * (I_mat - F_mat) ** 2

    return {
        "twcrps_discrete_sum": float(np.sum(score_mat)),
        "twcrps_discrete_mean_obs": float(np.sum(score_mat) / len(y)),
        "twcrps_discrete_mean_obs_thr": float(np.mean(score_mat)),
    }


def exp_standardize_egpd(
    y: np.ndarray,
    sigma: np.ndarray,
    kappa: np.ndarray,
    xi: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    F_y = egpd_cdf(y, sigma, kappa, xi)
    F_y = np.clip(F_y, eps, 1.0 - eps)

    z = -np.log(1.0 - F_y)

    return z


def smad_exponential_margins(
    y: np.ndarray,
    sigma: np.ndarray,
    kappa: np.ndarray,
    xi: np.ndarray,
    p1: float = 0.95,
) -> dict:
    z = exp_standardize_egpd(y, sigma, kappa, xi)
    z = z[np.isfinite(z)]

    n = len(z)
    if n < 10:
        return {
            "smad": np.nan,
            "smad_m": 0,
            "smad_p1": p1,
        }

    m = max(2, int(round((1.0 - p1) * n)))

    step = (1.0 - p1) / m
    p_grid = p1 + step * np.arange(m)
    p_grid = np.clip(p_grid, p1, 1.0 - step)

    q_emp = np.quantile(z, p_grid)
    q_exp = -np.log(1.0 - p_grid)

    smad = np.mean(np.abs(q_emp - q_exp))

    return {
        "smad": float(smad),
        "smad_m": int(m),
        "smad_p1": float(p1),
    }


def threshold_exceedance_summary(
    y,
    sigma,
    kappa,
    xi,
    thresholds=(0.5, 1.0, 2.0, 5.0, 10.0),
):
    out = {}
    y = np.asarray(y, dtype=float)

    for thr in thresholds:
        p_obs = np.mean(y > thr)
        p_pred = np.mean(1.0 - egpd_cdf(thr, sigma, kappa, xi))

        out[f"exc_obs_thr_{thr}"] = float(p_obs)
        out[f"exc_pred_thr_{thr}"] = float(p_pred)
        out[f"exc_abs_err_thr_{thr}"] = float(abs(p_obs - p_pred))

    return out


def summarize_crps_metrics(
    y: np.ndarray,
    sigma: np.ndarray,
    kappa: np.ndarray,
    xi: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> dict:
    y = np.asarray(y, dtype=float)
    y_pos = y[np.isfinite(y) & (y > 0)]

    if len(y_pos) == 0:
        raise ValueError("No positive rainfall values found.")

    out = {}

    if thresholds is None:
        thresholds = make_irregular_thresholds_from_data(y)

    crps_out = crps_integrated_egpd(
        y=y,
        sigma=sigma,
        kappa=kappa,
        xi=xi,
        thresholds=thresholds,
        weights=np.ones_like(thresholds, dtype=float),
    )

    out["crps_mean"] = crps_out["score_mean"]
    out["crps_sum"] = crps_out["score_sum"]

    tw_thresholds = make_rain_twcrps_thresholds(
        y=y,
        q_low=0.95,
        q_high=0.995,
        n_thresholds=24,
    )

    u0 = float(np.quantile(y_pos, 0.95))

    tw_weights = twcrps_weight_rain_power(
        thresholds=tw_thresholds,
        u0=u0,
        alpha=1.0,
        normalize=True,
    )

    tw_discrete = twcrps_discrete_sum_egpd(
        y=y,
        sigma=sigma,
        kappa=kappa,
        xi=xi,
        thresholds=tw_thresholds,
        weights=tw_weights,
    )

    out["twcrps_paper_sum"] = tw_discrete["twcrps_discrete_sum"]
    out["twcrps_paper_mean_obs"] = tw_discrete["twcrps_discrete_mean_obs"]
    out["twcrps_paper_mean_obs_thr"] = tw_discrete["twcrps_discrete_mean_obs_thr"]

    out["twcrps_threshold_q95"] = u0
    out["twcrps_n_thresholds"] = int(len(tw_thresholds))
    out["twcrps_threshold_min"] = float(np.min(tw_thresholds))
    out["twcrps_threshold_max"] = float(np.max(tw_thresholds))

    out["twcrps_mean"] = out["twcrps_paper_sum"]

    return out


def summarize_distribution_metrics(y, sigma, kappa, xi):
    out = {}
    y = np.asarray(y, dtype=float).reshape(-1)

    for p in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        q = qeGPD(p, sigma, kappa, xi)
        cover = (y <= q).mean()

        out[f"cover_{p:.2f}"] = float(cover)
        out[f"exc_{p:.2f}"] = float(1.0 - cover)
        out[f"abs_calib_err_{p:.2f}"] = float(abs(cover - p))

    pit = compute_pit(y, sigma, kappa, xi)

    out["pit_mean"] = float(np.mean(pit))
    out["pit_var"] = float(np.var(pit, ddof=1)) if len(pit) > 1 else np.nan
    out["pit_mean_err"] = float(abs(np.mean(pit) - 0.5))
    out["pit_var_err"] = (
        float(abs(np.var(pit, ddof=1) - 1.0 / 12.0))
        if len(pit) > 1
        else np.nan
    )

    out["mean_obs"] = float(np.mean(y))
    mean_pred_each = predicted_mean_mc(sigma, kappa, xi, n_mc=1000, seed=123)
    out["mean_pred"] = float(np.mean(mean_pred_each))
    out["mean_abs_err"] = float(abs(out["mean_pred"] - out["mean_obs"]))

    for p in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        q_obs = float(np.quantile(y, p))
        q_pred = float(np.median(qeGPD(p, sigma, kappa, xi)))

        out[f"qobs_{p:.2f}"] = q_obs
        out[f"qpred_{p:.2f}"] = q_pred
        out[f"q_abs_err_{p:.2f}"] = abs(q_obs - q_pred)

    out.update(
        threshold_exceedance_summary(
            y,
            sigma,
            kappa,
            xi,
            thresholds=(0.5, 1.0, 2.0, 5.0, 10.0),
        )
    )

    out.update(
        summarize_crps_metrics(
            y,
            sigma,
            kappa,
            xi,
        )
    )

    out.update(
        smad_exponential_margins(
            y=y,
            sigma=sigma,
            kappa=kappa,
            xi=xi,
            p1=0.95,
        )
    )

    return out


# %%
# Data preparation
def prepare_modeling_dataframe(
    df_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    df = df_raw.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)

    x_cols27 = get_x_cols27_downscaling(df)

    if len(x_cols27) == 0:
        raise ValueError("No radar columns matching X_pXX_dt... were found.")

    df = df.loc[df["Y_obs"].notna() & (df["Y_obs"] > 0)].copy()

    df[x_cols27] = df[x_cols27].apply(pd.to_numeric, errors="coerce")

    df["hour"] = df["time"].dt.hour.astype(int)
    df["minute"] = df["time"].dt.minute.astype(int)
    df["month"] = df["time"].dt.month.astype(int)
    df["year"] = df["time"].dt.year.astype(int)

    tod = df["hour"] * 60 + df["minute"]
    doy = df["time"].dt.dayofyear.astype(float)

    df["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    df["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    df["doy_sin"] = np.sin(2 * np.pi * (doy - 1) / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * (doy - 1) / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    X_block = df[x_cols27].to_numpy(dtype=float)

    df["radar_max"] = np.nanmax(X_block, axis=1)
    df["radar_mean"] = np.nanmean(X_block, axis=1)
    df["radar_sum"] = np.nansum(X_block, axis=1)

    x_cols_dt0h = [c for c in x_cols27 if c.endswith("dt0h")]

    if len(x_cols_dt0h) == 0:
        raise ValueError("No current-hour radar columns ending with dt0h were found.")

    X_block_dt0h = df[x_cols_dt0h].to_numpy(dtype=float)

    df["radar_max_dt0h"] = np.nanmax(X_block_dt0h, axis=1)
    df["radar_mean_dt0h"] = np.nanmean(X_block_dt0h, axis=1)
    df["radar_sum_dt0h"] = np.nansum(X_block_dt0h, axis=1)

    central_col = "X_p01_dt0h"

    if central_col in df.columns:
        df["radar_central_dt0h"] = df[central_col].astype(float)
    else:
        print(f"Warning: {central_col} not found. Using radar_max_dt0h instead.")
        df["radar_central_dt0h"] = df["radar_max_dt0h"]

    df["corres"] = (
        (df["Y_obs"] > 0)
        & (np.nansum(X_block, axis=1) > 0)
    ).astype(int)

    keep_cols = [
        "time",
        "station",
        "Y_obs",
        "year",
        "month",
        "hour",
        "minute",
        *TIME_COLS,
        *SPATIAL_COLS,
        "radar_max",
        "radar_mean",
        "radar_sum",
        "radar_max_dt0h",
        "radar_mean_dt0h",
        "radar_sum_dt0h",
        "radar_central_dt0h",
        *x_cols27,
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df.loc[df["corres"] == 1, keep_cols].copy()

    x_cols_all = list(dict.fromkeys(
        TIME_COLS
        + SPATIAL_COLS
        + [
            "radar_max",
            "radar_mean",
            "radar_sum",
            "radar_max_dt0h",
            "radar_mean_dt0h",
            "radar_sum_dt0h",
            "radar_central_dt0h",
        ]
        + x_cols27
    ))

    x_cols_all = [c for c in x_cols_all if c in df.columns]

    return df, x_cols27, x_cols_dt0h, x_cols_all


def make_covariate_sets(x_cols27, x_cols_dt0h, x_cols_all):
    radar_summary_all = [
        "radar_max",
        "radar_mean",
        "radar_sum",
        "radar_max_dt0h",
        "radar_mean_dt0h",
        "radar_sum_dt0h",
        "radar_central_dt0h",
    ]

    radar_summary_all = [c for c in radar_summary_all if c in x_cols_all]

    local_pixels = [
        c for c in [
            "X_p01_dt0h",
            "X_p02_dt0h",
            "X_p03_dt0h",
            "X_p04_dt0h",
            "X_p05_dt0h",
        ]
        if c in x_cols_all
    ]

    x_sets = {
        "central_only": ["radar_central_dt0h"],
        "summary_dt0h": [
            "radar_max_dt0h",
            "radar_mean_dt0h",
            "radar_sum_dt0h",
            "radar_central_dt0h",
        ],
        "local_pixels_dt0h": local_pixels,
        "all_pixels_dt0h": x_cols_dt0h,
        "radar_summaries": radar_summary_all,
        "radar_all": radar_summary_all + x_cols27,
        "radar_time": TIME_COLS + radar_summary_all + x_cols27,
        "radar_time_space": x_cols_all,
    }

    x_sets = {
        name: list(dict.fromkeys([c for c in cols if c in x_cols_all]))
        for name, cols in x_sets.items()
    }

    x_sets = {
        name: cols
        for name, cols in x_sets.items()
        if len(cols) > 0
    }

    return x_sets


# %%
# Standardization + XY builders

def standardize_train_only(
    df: pd.DataFrame,
    train_idx_labels: np.ndarray,
    scale_cols: list[str],
):
    df2 = df.copy()
    df2[scale_cols] = df2[scale_cols].astype(float)

    mu = df2.loc[train_idx_labels, scale_cols].mean(axis=0, skipna=True)
    sdv = (
        df2.loc[train_idx_labels, scale_cols]
        .std(axis=0, skipna=True, ddof=1)
        .replace(0.0, 1.0)
    )

    df2[scale_cols] = (df2[scale_cols] - mu) / sdv

    return df2, mu, sdv


def build_xy_train_valid(
    df_std: pd.DataFrame,
    x_cols: list[str],
    train_idx_labels,
    valid_idx_labels,
):
    X = df_std[x_cols].to_numpy(dtype=np.float32)
    Y = df_std["Y_obs"].to_numpy(dtype=np.float32)

    X_all = X.reshape(len(X), 1, X.shape[1])

    tr_pos = df_std.index.get_indexer(train_idx_labels)
    va_pos = df_std.index.get_indexer(valid_idx_labels)

    if (tr_pos < 0).any() or (va_pos < 0).any():
        raise ValueError("Split indices not found in df_std.index.")

    return {
        "X_all": X_all,
        "Y_all": Y,
        "tr_pos": tr_pos,
        "va_pos": va_pos,
        "X_train": X_all[tr_pos],
        "Y_train": Y[tr_pos].reshape(-1, 1),
        "X_valid": X_all[va_pos],
        "Y_valid": Y[va_pos].reshape(-1, 1),
    }


def build_X_from_meta(df: pd.DataFrame, meta: dict) -> np.ndarray:
    df2 = df.copy()
    x_cols = meta["x_cols"]

    df2[x_cols] = df2[x_cols].astype(float)
    df2[x_cols] = (
        df2[x_cols]
        - meta["mu"][x_cols]
    ) / meta["sdv"][x_cols].replace(0.0, 1.0)

    X = df2[x_cols].to_numpy(np.float32)

    return X.reshape(len(df2), 1, -1)


# %%
# Model 1: fit EGPD
def fit_egpd_stationary_direct(
    y_train,
    y_valid=None,
    sigma_init=0.57,
    kappa_init=0.28,
    xi_init=0.24,
    fix_xi=True,
    censor_threshold=0.22,
):
    y_train = np.asarray(y_train, dtype=float).reshape(-1)

    if fix_xi:
        theta0 = np.array(
            [
                np.log(sigma_init),
                np.log(kappa_init),
            ],
            dtype=float,
        )

        def objective(theta):
            log_sigma, log_kappa = theta

            sigma = np.exp(log_sigma) * np.ones_like(y_train)
            kappa = np.exp(log_kappa) * np.ones_like(y_train)
            xi = np.full_like(y_train, xi_init, dtype=float)

            return egpd_left_censored_nll(
                y=y_train,
                sigma=sigma,
                kappa=kappa,
                xi=xi,
                c=censor_threshold,
            )

        res = minimize(objective, theta0, method="L-BFGS-B")

        log_sigma_hat, log_kappa_hat = res.x

        sigma_hat = float(np.exp(log_sigma_hat))
        kappa_hat = float(np.exp(log_kappa_hat))
        xi_hat = float(xi_init)

    else:
        theta0 = np.array(
            [
                np.log(sigma_init),
                np.log(kappa_init),
                math.log(xi_init / (1.0 - xi_init)),
            ],
            dtype=float,
        )

        def objective(theta):
            log_sigma, log_kappa, xi_logit = theta

            sigma = np.exp(log_sigma) * np.ones_like(y_train)
            kappa = np.exp(log_kappa) * np.ones_like(y_train)
            xi_scalar = 1.0 / (1.0 + np.exp(-xi_logit))
            xi = np.full_like(y_train, xi_scalar, dtype=float)

            return egpd_left_censored_nll(
                y=y_train,
                sigma=sigma,
                kappa=kappa,
                xi=xi,
                c=censor_threshold,
            )

        res = minimize(objective, theta0, method="L-BFGS-B")

        log_sigma_hat, log_kappa_hat, xi_logit_hat = res.x

        sigma_hat = float(np.exp(log_sigma_hat))
        kappa_hat = float(np.exp(log_kappa_hat))
        xi_hat = float(1.0 / (1.0 + np.exp(-xi_logit_hat)))

    sigma_train = np.full_like(y_train, sigma_hat, dtype=float)
    kappa_train = np.full_like(y_train, kappa_hat, dtype=float)
    xi_train = np.full_like(y_train, xi_hat, dtype=float)

    out = {
        "success": bool(res.success),
        "message": res.message,
        "sigma_hat": sigma_hat,
        "kappa_hat": kappa_hat,
        "xi_hat": xi_hat,
        "train_nll": egpd_left_censored_nll(
            y_train,
            sigma_train,
            kappa_train,
            xi_train,
            c=censor_threshold,
        ),
        "train_nll_sum": egpd_left_censored_nll_sum(
            y_train,
            sigma_train,
            kappa_train,
            xi_train,
            c=censor_threshold,
        ),
    }

    if y_valid is not None:
        y_valid = np.asarray(y_valid, dtype=float).reshape(-1)

        sigma_valid = np.full_like(y_valid, sigma_hat, dtype=float)
        kappa_valid = np.full_like(y_valid, kappa_hat, dtype=float)
        xi_valid = np.full_like(y_valid, xi_hat, dtype=float)

        out["val_nll"] = egpd_left_censored_nll(
            y_valid,
            sigma_valid,
            kappa_valid,
            xi_valid,
            c=censor_threshold,
        )
        out["val_nll_sum"] = egpd_left_censored_nll_sum(
            y_valid,
            sigma_valid,
            kappa_valid,
            xi_valid,
            c=censor_threshold,
        )

    return out


# %%
# GLM / GAM
def check_variant(variant: str):
    if variant not in ALLOWED_VARIANTS:
        raise ValueError(f"variant must be one of {ALLOWED_VARIANTS}, got {variant}")


def make_bspline_design(x, n_inner_knots=4, degree=3):
    x = np.asarray(x, dtype=float).reshape(-1)

    xmin, xmax = np.min(x), np.max(x)

    if np.isclose(xmin, xmax):
        return np.ones((len(x), 1), dtype=float), {
            "knots": None,
            "degree": degree,
        }

    inner = np.linspace(xmin, xmax, n_inner_knots + 2)[1:-1]

    knots = np.concatenate([
        np.repeat(xmin, degree + 1),
        inner,
        np.repeat(xmax, degree + 1),
    ])

    n_basis = len(knots) - degree - 1
    B = np.zeros((len(x), n_basis), dtype=float)

    for j in range(n_basis):
        coef = np.zeros(n_basis)
        coef[j] = 1.0

        spl = BSpline(knots, coef, degree, extrapolate=True)
        B[:, j] = spl(x)

    return B, {
        "knots": knots,
        "degree": degree,
    }


def transform_params_from_theta(theta, X_sigma, X_kappa, variant, xi_fixed):
    n = X_sigma.shape[0] if X_sigma is not None else X_kappa.shape[0]
    i = 0

    if variant in {"both", "sigma_only"}:
        p_s = X_sigma.shape[1]
        beta_s = theta[i:i + p_s]
        i += p_s

        log_sigma = X_sigma @ beta_s
        sigma = np.exp(log_sigma)

    else:
        log_sigma0 = theta[i]
        i += 1

        sigma = np.exp(log_sigma0) * np.ones(n)

    if variant in {"both", "kappa_only"}:
        p_k = X_kappa.shape[1]
        beta_k = theta[i:i + p_k]
        i += p_k

        log_kappa = X_kappa @ beta_k
        kappa = np.exp(log_kappa)

    else:
        log_kappa0 = theta[i]
        i += 1

        kappa = np.exp(log_kappa0) * np.ones(n)

    xi = np.full(n, xi_fixed, dtype=float)

    return sigma, kappa, xi


def fit_egpd_regression_model(
    y_train,
    x_train,
    model_type="glm",
    variant="both",
    xi_fixed=0.24,
    sigma_init=0.57,
    kappa_init=0.28,
    censor_threshold=0.22,
    lambda_ridge=1e-4,
    lambda_smooth=1e-3,
    n_inner_knots=4,
):
    check_variant(variant)

    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    x_train = np.asarray(x_train, dtype=float).reshape(-1)

    if model_type == "glm":
        x_mean = np.mean(x_train)
        x_sd = np.std(x_train, ddof=1)

        if x_sd <= 0:
            x_sd = 1.0

        z_train = (x_train - x_mean) / x_sd
        X_base_train = np.column_stack([np.ones(len(z_train)), z_train])

        basis_meta = {
            "type": "glm",
            "x_mean": x_mean,
            "x_sd": x_sd,
        }

    elif model_type == "gam":
        X_base_train, bs_meta = make_bspline_design(
            x_train,
            n_inner_knots=n_inner_knots,
            degree=3,
        )

        basis_meta = {
            "type": "gam",
            **bs_meta,
        }

    else:
        raise ValueError("model_type must be 'glm' or 'gam'.")

    X_sigma_train = X_base_train if variant in {"both", "sigma_only"} else None
    X_kappa_train = X_base_train if variant in {"both", "kappa_only"} else None

    if variant == "both":
        theta0 = np.concatenate([
            np.zeros(X_base_train.shape[1]),
            np.zeros(X_base_train.shape[1]),
        ])

        theta0[0] = np.log(sigma_init)
        theta0[X_base_train.shape[1]] = np.log(kappa_init)

    elif variant == "sigma_only":
        theta0 = np.concatenate([
            np.zeros(X_base_train.shape[1]),
            np.array([np.log(kappa_init)]),
        ])

        theta0[0] = np.log(sigma_init)

    elif variant == "kappa_only":
        theta0 = np.concatenate([
            np.array([np.log(sigma_init)]),
            np.zeros(X_base_train.shape[1]),
        ])

        theta0[1] = np.log(kappa_init)

    def objective(theta):
        sigma, kappa, xi = transform_params_from_theta(
            theta=theta,
            X_sigma=X_sigma_train,
            X_kappa=X_kappa_train,
            variant=variant,
            xi_fixed=xi_fixed,
        )

        nll = egpd_left_censored_nll(
            y=y_train,
            sigma=sigma,
            kappa=kappa,
            xi=xi,
            c=censor_threshold,
        )

        pen = lambda_ridge * np.sum(theta ** 2)

        if model_type == "gam":
            if variant == "both":
                p = X_base_train.shape[1]
                beta_s = theta[:p]
                beta_k = theta[p:(2 * p)]

                pen += lambda_smooth * np.sum(np.diff(beta_s, n=2) ** 2)
                pen += lambda_smooth * np.sum(np.diff(beta_k, n=2) ** 2)

            elif variant == "sigma_only":
                p = X_base_train.shape[1]
                beta_s = theta[:p]

                pen += lambda_smooth * np.sum(np.diff(beta_s, n=2) ** 2)

            elif variant == "kappa_only":
                p = X_base_train.shape[1]
                beta_k = theta[1:(1 + p)]

                pen += lambda_smooth * np.sum(np.diff(beta_k, n=2) ** 2)

        return nll + pen

    res = minimize(objective, theta0, method="L-BFGS-B")

    theta_hat = res.x

    sigma_train, kappa_train, xi_train = transform_params_from_theta(
        theta_hat,
        X_sigma_train,
        X_kappa_train,
        variant,
        xi_fixed,
    )

    out = {
        "success": bool(res.success),
        "message": res.message,
        "theta_hat": theta_hat,
        "basis_meta": basis_meta,
        "variant": variant,
        "model_type": model_type,
        "train_nll": egpd_left_censored_nll(
            y_train,
            sigma_train,
            kappa_train,
            xi_train,
            c=censor_threshold,
        ),
        "train_nll_sum": egpd_left_censored_nll_sum(
            y_train,
            sigma_train,
            kappa_train,
            xi_train,
            c=censor_threshold,
        ),
    }

    return out


def predict_egpd_regression_model(fit, x_new, xi_fixed):
    x_new = np.asarray(x_new, dtype=float).reshape(-1)

    model_type = fit["model_type"]
    variant = fit["variant"]
    theta_hat = fit["theta_hat"]
    basis_meta = fit["basis_meta"]

    if model_type == "glm":
        z = (x_new - basis_meta["x_mean"]) / basis_meta["x_sd"]
        X_base = np.column_stack([np.ones(len(z)), z])

    else:
        if basis_meta["knots"] is None:
            X_base = np.ones((len(x_new), 1), dtype=float)

        else:
            knots = basis_meta["knots"]
            degree = basis_meta["degree"]
            n_basis = len(knots) - degree - 1

            X_base = np.zeros((len(x_new), n_basis), dtype=float)

            for j in range(n_basis):
                coef = np.zeros(n_basis)
                coef[j] = 1.0

                spl = BSpline(knots, coef, degree, extrapolate=True)
                X_base[:, j] = spl(x_new)

    X_sigma = X_base if variant in {"both", "sigma_only"} else None
    X_kappa = X_base if variant in {"both", "kappa_only"} else None

    sigma, kappa, xi = transform_params_from_theta(
        theta_hat,
        X_sigma,
        X_kappa,
        variant,
        xi_fixed=xi_fixed,
    )

    return {
        "pred_sigma": sigma,
        "pred_kappa": kappa,
        "pred_xi": xi,
    }


def get_gam_initial_values(
    df_train: pd.DataFrame,
    covariate_col: str,
    variant: str = "both",
    xi_fixed: float = XI_INIT,
    sigma_init: float = SIGMA_INIT,
    kappa_init: float = KAPPA_INIT,
    censor_threshold: float = CENSOR_THRESHOLD,
) -> dict:
    """
    Fit a one-covariate GAM on the training data and use the median fitted
    sigma/kappa as scalar NN initial values.

    This does not initialize NN weights with the GAM function.
    It initializes the distribution level around a meaningful GAM solution.
    """
    y_train = df_train["Y_obs"].to_numpy(float)
    x_train = df_train[covariate_col].to_numpy(float)

    fit_gam = fit_egpd_regression_model(
        y_train=y_train,
        x_train=x_train,
        model_type="gam",
        variant=variant,
        xi_fixed=xi_fixed,
        sigma_init=sigma_init,
        kappa_init=kappa_init,
        censor_threshold=censor_threshold,
    )

    pred_train = predict_egpd_regression_model(
        fit_gam,
        x_train,
        xi_fixed=xi_fixed,
    )

    sigma_med = float(np.nanmedian(pred_train["pred_sigma"]))
    kappa_med = float(np.nanmedian(pred_train["pred_kappa"]))

    sigma_med = float(np.clip(sigma_med, 1e-3, 100.0))
    kappa_med = float(np.clip(kappa_med, 5e-2, 50.0))

    return {
        "sigma_init_gam": sigma_med,
        "kappa_init_gam": kappa_med,
        "gam_success": bool(fit_gam["success"]),
        "gam_train_nll": float(fit_gam["train_nll"]),
    }


# %%
# NN model
@dataclass
class Config:
    name: str
    widths: Tuple[int, ...] = (16, 8)


def build_X_meta_for_variant(df_model, split, variant, x_cols):
    df_std, mu, sdv = standardize_train_only(
        df_model,
        split["train_idx"],
        x_cols,
    )

    built = build_xy_train_valid(
        df_std,
        x_cols,
        split["train_idx"],
        split["valid_idx"],
    )

    meta = {
        "x_cols": x_cols,
        "mu": mu,
        "sdv": sdv,
        "variant": variant,
    }

    return df_std, built, meta


def predict_params_on_df_variant(df: pd.DataFrame, fit: dict, meta: dict):
    X = build_X_from_meta(df, meta)
    variant = meta["variant"]

    if variant == "both":
        out = predict_params(fit["model"], X_s=X, X_k=X, offset=None)

    elif variant == "sigma_only":
        out = predict_params(fit["model"], X_s=X, X_k=None, offset=None)

    elif variant == "kappa_only":
        out = predict_params(fit["model"], X_s=None, X_k=X, offset=None)

    else:
        raise ValueError("Unknown variant.")

    return (
        out["pred_sigma"].reshape(-1),
        out["pred_kappa"].reshape(-1),
        out["pred_xi"].reshape(-1),
    )


def run_one_nn_variant(
    df_model: pd.DataFrame,
    split: Dict[str, Any],
    x_cols: list[str],
    cfg: Config,
    variant: str,
    sigma_init: float,
    kappa_init: float,
    xi_init: float,
    lr: float = 1e-4,
    n_ep: int = 200,
    batch_size: int = 64,
    seed: int = 1,
    device: Optional[str] = None,
    censor_threshold: float = 0.22,
    early_stopping: bool = True,
    patience: int = 25,
    warmup_epochs: int = 20,
    weight_decay: float = 0.0,
):
    check_variant(variant)

    df_std, built, meta = build_X_meta_for_variant(
        df_model,
        split,
        variant,
        x_cols,
    )

    p = built["X_train"].shape[-1]

    d_s = p if variant in {"both", "sigma_only"} else None
    d_k = p if variant in {"both", "kappa_only"} else None

    model = EGPDPINN_NNOnly(
        d_s=d_s,
        d_k=d_k,
        init_scale=sigma_init,
        init_kappa=kappa_init,
        init_xi=xi_init,
        widths=parse_widths(cfg.widths),
    )

    Xs_train = built["X_train"] if variant in {"both", "sigma_only"} else None
    Xk_train = built["X_train"] if variant in {"both", "kappa_only"} else None
    Xs_valid = built["X_valid"] if variant in {"both", "sigma_only"} else None
    Xk_valid = built["X_valid"] if variant in {"both", "kappa_only"} else None

    train_kwargs = dict(
        model=model,
        X_s=Xs_train,
        X_k=Xk_train,
        Y_train=built["Y_train"],
        X_s_valid=Xs_valid,
        X_k_valid=Xk_valid,
        Y_valid=built["Y_valid"],
        offset=None,
        offset_valid=None,
        n_epochs=n_ep,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
        device=device,
        early_stopping=early_stopping,
        patience=patience,
        min_delta=0.0,
        warmup_epochs=warmup_epochs,
        censor_threshold=censor_threshold,
    )

    sig = inspect.signature(train_egpd_nn_only)
    if "weight_decay" in sig.parameters:
        train_kwargs["weight_decay"] = weight_decay
    else:
        if weight_decay != 0.0:
            print(
                "Warning: train_egpd_nn_only does not accept weight_decay. "
                "Ignoring weight_decay."
            )

    fit = train_egpd_nn_only(**train_kwargs)

    meta["cfg"] = cfg

    return fit, meta


def evaluate_nn_config_on_split(
    df_model: pd.DataFrame,
    split: Dict[str, Any],
    x_cols: list[str],
    cfg: Config,
    variant: str,
    sigma_init: float,
    kappa_init: float,
    xi_init: float,
    batch_size: int,
    censor_threshold: float,
    lr: float,
    n_ep: int,
    seed: int = 1,
    device: Optional[str] = None,
    weight_decay: float = 0.0,
):
    fit, meta = run_one_nn_variant(
        df_model=df_model,
        split=split,
        x_cols=x_cols,
        cfg=cfg,
        variant=variant,
        sigma_init=sigma_init,
        kappa_init=kappa_init,
        xi_init=xi_init,
        lr=lr,
        n_ep=n_ep,
        batch_size=batch_size,
        seed=seed,
        device=device,
        censor_threshold=censor_threshold,
        early_stopping=True,
        patience=25,
        warmup_epochs=20,
        weight_decay=weight_decay,
    )

    df_valid = df_model.loc[split["valid_idx"]].copy()
    y_valid = df_valid["Y_obs"].to_numpy(np.float32)

    sigma_v, kappa_v, xi_v = predict_params_on_df_variant(
        df_valid,
        fit,
        meta,
    )

    metrics = summarize_distribution_metrics(
        y_valid,
        sigma_v,
        kappa_v,
        xi_v,
    )

    out = {
        "valid_loss": float(fit.get("val_nll", np.nan)),
        "valid_loss_sum": egpd_left_censored_nll_sum(
            y_valid,
            sigma_v,
            kappa_v,
            xi_v,
            c=censor_threshold,
        ),
        "train_loss": float(fit["train_nll"]),
        "stopped_epoch": int(fit.get("stopped_epoch", n_ep)),
        "crps_mean": metrics["crps_mean"],
        "crps_sum": metrics["crps_sum"],
        "twcrps_mean": metrics["twcrps_mean"],
        "twcrps_paper_sum": metrics["twcrps_paper_sum"],
        "twcrps_paper_mean_obs": metrics["twcrps_paper_mean_obs"],
        "twcrps_paper_mean_obs_thr": metrics["twcrps_paper_mean_obs_thr"],
        "smad": metrics["smad"],
        "smad_m": metrics["smad_m"],
        "err95": metrics["abs_calib_err_0.95"],
        "err99": metrics["abs_calib_err_0.99"],
    }

    return out


def evaluate_nn_config_on_split_fast(
    df_model: pd.DataFrame,
    split: Dict[str, Any],
    x_cols: list[str],
    cfg: Config,
    variant: str,
    sigma_init: float,
    kappa_init: float,
    xi_init: float,
    batch_size: int,
    censor_threshold: float,
    lr: float,
    n_ep: int,
    seed: int = 1,
    device: Optional[str] = None,
    weight_decay: float = 0.0,
):
    fit, meta = run_one_nn_variant(
        df_model=df_model,
        split=split,
        x_cols=x_cols,
        cfg=cfg,
        variant=variant,
        sigma_init=sigma_init,
        kappa_init=kappa_init,
        xi_init=xi_init,
        lr=lr,
        n_ep=n_ep,
        batch_size=batch_size,
        seed=seed,
        device=device,
        censor_threshold=censor_threshold,
        early_stopping=True,
        patience=10,
        warmup_epochs=10,
        weight_decay=weight_decay,
    )

    return {
        "valid_loss": float(fit.get("val_nll", np.nan)),
        "train_loss": float(fit["train_nll"]),
        "stopped_epoch": int(fit.get("stopped_epoch", n_ep)),
    }


def tune_nn_on_outer_train(
    df_model: pd.DataFrame,
    outer_split: Dict[str, Any],
    x_sets: dict[str, list[str]],
    param_grid: dict,
    seed: int = 1,
    device: Optional[str] = None,
):
    df_outer_train = df_model.loc[outer_split["train_idx"]].copy()

    inner_split = make_single_split_from_train(
        df_outer_train,
        train_frac=0.8,
        block="30D",
        seed=seed + 100,
    )

    rows = []

    common_keys = [
        "variant",
        "x_set_name",
        "widths",
        "lr",
        "weight_decay",
        "batch_size",
        "n_ep",
        "xi_init",
        "censor_threshold",
        "init_source",
    ]

    common_values = [param_grid[k] for k in common_keys]

    for vals in product(*common_values):
        params = dict(zip(common_keys, vals))

        variant = params["variant"]
        x_set_name = params["x_set_name"]
        init_source = params["init_source"]

        if x_set_name not in x_sets:
            raise ValueError(f"Unknown x_set_name: {x_set_name}")

        x_cols = x_sets[x_set_name]

        if variant == "both":
            sigma_candidates = param_grid["sigma_init"]
            kappa_candidates = param_grid["kappa_init"]

        elif variant == "sigma_only":
            sigma_candidates = param_grid["sigma_init"]
            kappa_candidates = [KAPPA_INIT]

        elif variant == "kappa_only":
            sigma_candidates = [SIGMA_INIT]
            kappa_candidates = param_grid["kappa_init"]

        else:
            raise ValueError(f"Unknown variant: {variant}")

        gam_init = None

        if init_source == "gam":
            try:
                gam_init = get_gam_initial_values(
                    df_train=df_outer_train.loc[inner_split["train_idx"]],
                    covariate_col=single_cov_col,
                    variant=variant,
                    xi_fixed=float(params["xi_init"]),
                    sigma_init=SIGMA_INIT,
                    kappa_init=KAPPA_INIT,
                    censor_threshold=float(params["censor_threshold"]),
                )

                sigma_candidates = [gam_init["sigma_init_gam"]]
                kappa_candidates = [gam_init["kappa_init_gam"]]

            except Exception as e:
                print(f"GAM init failed for {params}: {e}")
                continue

        for sigma_init, kappa_init in product(sigma_candidates, kappa_candidates):
            params_variant = params.copy()

            params_variant["sigma_init"] = float(sigma_init)
            params_variant["kappa_init"] = float(kappa_init)
            params_variant["n_covariates"] = len(x_cols)

            if gam_init is not None:
                params_variant.update(gam_init)

            cfg = Config(
                name=f"nn_{x_set_name}_{variant}_{params_variant['widths']}_{init_source}",
                widths=parse_widths(params_variant["widths"]),
            )

            res = evaluate_nn_config_on_split_fast(
                df_model=df_model,
                split=inner_split,
                x_cols=x_cols,
                cfg=cfg,
                variant=params_variant["variant"],
                sigma_init=float(params_variant["sigma_init"]),
                kappa_init=float(params_variant["kappa_init"]),
                xi_init=float(params_variant["xi_init"]),
                batch_size=int(params_variant["batch_size"]),
                censor_threshold=float(params_variant["censor_threshold"]),
                lr=float(params_variant["lr"]),
                n_ep=int(params_variant["n_ep"]),
                seed=seed,
                device=device,
                weight_decay=float(params_variant["weight_decay"]),
            )

            row = params_variant.copy()
            row.update(res)
            rows.append(row)

            print(
                f"tested x_set={x_set_name:18s} | "
                f"init={init_source:8s} | "
                f"variant={variant:10s} | "
                f"widths={params_variant['widths']} | "
                f"n_cov={len(x_cols):2d} | "
                f"valid_loss={res['valid_loss']:.4f} | "
                f"train_loss={res['train_loss']:.4f} | "
                f"stopped_epoch={res['stopped_epoch']}"
            )

    tuning_df = pd.DataFrame(rows)

    if len(tuning_df) == 0:
        raise RuntimeError("No NN configuration was successfully evaluated.")

    tuning_df = tuning_df.sort_values(
        [
            "valid_loss",
            "train_loss",
            "n_covariates",
            "stopped_epoch",
        ]
    ).reset_index(drop=True)

    best_params = tuning_df.iloc[0].to_dict()

    return tuning_df, best_params


def rerank_top_nn_configs(
    df_model: pd.DataFrame,
    outer_split: Dict[str, Any],
    x_sets: dict[str, list[str]],
    tuning_df: pd.DataFrame,
    top_k: int = 10,
    seed: int = 1,
    device: Optional[str] = None,
):
    df_outer_train = df_model.loc[outer_split["train_idx"]].copy()

    inner_split = make_single_split_from_train(
        df_outer_train,
        train_frac=0.8,
        block="30D",
        seed=seed + 100,
    )

    rows = []

    top_configs = tuning_df.head(top_k).copy()

    for _, params in top_configs.iterrows():
        params = params.to_dict()

        x_set_name = params["x_set_name"]
        x_cols = x_sets[x_set_name]

        cfg = Config(
            name=f"nn_rerank_{x_set_name}_{params['variant']}_{params['widths']}",
            widths=parse_widths(params["widths"]),
        )

        res = evaluate_nn_config_on_split(
            df_model=df_model,
            split=inner_split,
            x_cols=x_cols,
            cfg=cfg,
            variant=params["variant"],
            sigma_init=float(params["sigma_init"]),
            kappa_init=float(params["kappa_init"]),
            xi_init=float(params["xi_init"]),
            batch_size=int(params["batch_size"]),
            censor_threshold=float(params["censor_threshold"]),
            lr=float(params["lr"]),
            n_ep=int(params["n_ep"]),
            seed=seed,
            device=device,
            weight_decay=float(params.get("weight_decay", 0.0)),
        )

        row = params.copy()
        row.update(res)
        rows.append(row)

        print(
            f"reranked x_set={x_set_name:18s} | "
            f"init={params.get('init_source', 'unknown'):8s} | "
            f"variant={params['variant']:10s} | "
            f"valid_loss={res['valid_loss']:.4f} | "
            f"twCRPS_sum={res['twcrps_paper_sum']:.4f} | "
            f"sMAD={res['smad']:.4f} | "
            f"CRPS={res['crps_mean']:.4f} | "
            f"err95={res['err95']:.4f} | "
            f"err99={res['err99']:.4f}"
        )

    rerank_df = pd.DataFrame(rows)

    rerank_df["rank_valid_loss"] = rerank_df["valid_loss"].rank(method="average")
    rerank_df["rank_twcrps"] = rerank_df["twcrps_paper_sum"].rank(method="average")
    rerank_df["rank_smad"] = rerank_df["smad"].rank(method="average")
    rerank_df["rank_crps"] = rerank_df["crps_mean"].rank(method="average")
    rerank_df["rank_err95"] = rerank_df["err95"].rank(method="average")
    rerank_df["rank_err99"] = rerank_df["err99"].rank(method="average")

    rerank_df = rerank_df.sort_values(
        [
            "valid_loss",
            "twcrps_paper_sum",
            "crps_mean",
            "smad",
        ]
    ).reset_index(drop=True)

    best_params = rerank_df.iloc[0].to_dict()

    return rerank_df, best_params


def evaluate_fixed_nn_model(
    df_model: pd.DataFrame,
    splits: list[dict],
    x_sets: dict[str, list[str]],
    best_params: dict,
    seed: int = 1,
    device: Optional[str] = None,
):
    rows = []

    x_set_name = best_params["x_set_name"]
    x_cols = x_sets[x_set_name]

    cfg = Config(
        name=f"nn_tuned_{x_set_name}",
        widths=parse_widths(best_params["widths"]),
    )

    for sp in splits:
        fit, meta = run_one_nn_variant(
            df_model=df_model,
            split=sp,
            x_cols=x_cols,
            cfg=cfg,
            variant=best_params["variant"],
            sigma_init=float(best_params["sigma_init"]),
            kappa_init=float(best_params["kappa_init"]),
            xi_init=float(best_params["xi_init"]),
            lr=float(best_params["lr"]),
            n_ep=int(best_params["n_ep"]),
            batch_size=int(best_params["batch_size"]),
            seed=seed + sp["fold"],
            device=device,
            censor_threshold=float(best_params["censor_threshold"]),
            early_stopping=True,
            patience=25,
            warmup_epochs=20,
            weight_decay=float(best_params.get("weight_decay", 0.0)),
        )

        df_valid = df_model.loc[sp["valid_idx"]].copy()
        y_valid = df_valid["Y_obs"].to_numpy(np.float32)

        sigma_v, kappa_v, xi_v = predict_params_on_df_variant(
            df_valid,
            fit,
            meta,
        )

        metrics = summarize_distribution_metrics(
            y_valid,
            sigma_v,
            kappa_v,
            xi_v,
        )

        sigma_all, kappa_all, xi_all = predict_params_on_df_variant(
            df_model,
            fit,
            meta,
        )

        y_all = df_model["Y_obs"].to_numpy(float)

        smad_all = smad_exponential_margins(
            y=y_all,
            sigma=sigma_all,
            kappa=kappa_all,
            xi=xi_all,
            p1=0.95,
        )

        row = {
            "fold": sp["fold"],
            "model_family": "nn",
            "model_type": "nn_tuned",
            "variant": best_params["variant"],
            "covariate": x_set_name,
            "train_loss": float(fit["train_nll"]),
            "valid_loss": float(fit.get("val_nll", np.nan)),
            "valid_loss_sum": egpd_left_censored_nll_sum(
                y_valid,
                sigma_v,
                kappa_v,
                xi_v,
                c=float(best_params["censor_threshold"]),
            ),
            "n_train": len(sp["train_idx"]),
            "n_valid": len(sp["valid_idx"]),
            "best_x_set_name": x_set_name,
            "best_n_covariates": len(x_cols),
            "best_widths": str(best_params["widths"]),
            "best_lr": float(best_params["lr"]),
            "best_weight_decay": float(best_params.get("weight_decay", 0.0)),
            "best_batch_size": int(best_params["batch_size"]),
            "best_n_ep": int(best_params["n_ep"]),
            "best_sigma_init": float(best_params["sigma_init"]),
            "best_kappa_init": float(best_params["kappa_init"]),
            "best_xi_init": float(best_params["xi_init"]),
            "best_censor_threshold": float(best_params["censor_threshold"]),
            "best_init_source": best_params.get("init_source", "unknown"),
            "stopped_epoch": int(fit.get("stopped_epoch", np.nan)),
            "smad_original": smad_all["smad"],
            "smad_original_m": smad_all["smad_m"],
        }

        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


# %%
# Evaluation
def evaluate_stationary_candidate(
    df_model: pd.DataFrame,
    splits: list[dict],
    sigma_init: float,
    kappa_init: float,
    xi_init: float,
    censor_threshold: float,
    fix_xi: bool = True,
):
    rows = []

    for sp in splits:
        y_train = df_model.loc[sp["train_idx"], "Y_obs"].to_numpy(float)
        y_valid = df_model.loc[sp["valid_idx"], "Y_obs"].to_numpy(float)

        fit = fit_egpd_stationary_direct(
            y_train=y_train,
            y_valid=y_valid,
            sigma_init=sigma_init,
            kappa_init=kappa_init,
            xi_init=xi_init,
            fix_xi=fix_xi,
            censor_threshold=censor_threshold,
        )

        sigma_v = np.full(len(y_valid), fit["sigma_hat"], dtype=float)
        kappa_v = np.full(len(y_valid), fit["kappa_hat"], dtype=float)
        xi_v = np.full(len(y_valid), fit["xi_hat"], dtype=float)

        metrics = summarize_distribution_metrics(
            y_valid,
            sigma_v,
            kappa_v,
            xi_v,
        )

        y_all = df_model["Y_obs"].to_numpy(float)
        sigma_all = np.full(len(y_all), fit["sigma_hat"], dtype=float)
        kappa_all = np.full(len(y_all), fit["kappa_hat"], dtype=float)
        xi_all = np.full(len(y_all), fit["xi_hat"], dtype=float)

        smad_all = smad_exponential_margins(
            y=y_all,
            sigma=sigma_all,
            kappa=kappa_all,
            xi=xi_all,
            p1=0.95,
        )

        row = {
            "fold": sp["fold"],
            "model_family": "stationary",
            "model_type": "stationary",
            "variant": "stationary",
            "covariate": "none",
            "train_loss": fit["train_nll"],
            "valid_loss": fit.get("val_nll", np.nan),
            "valid_loss_sum": fit.get("val_nll_sum", np.nan),
            "n_train": len(y_train),
            "n_valid": len(y_valid),
            "sigma_hat": fit["sigma_hat"],
            "kappa_hat": fit["kappa_hat"],
            "xi_hat": fit["xi_hat"],
            "success": fit["success"],
            "smad_original": smad_all["smad"],
            "smad_original_m": smad_all["smad_m"],
        }

        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_single_covariate_model(
    df_model: pd.DataFrame,
    splits: list[dict],
    model_type: str,
    variant: str,
    covariate_col: str,
    sigma_init: float,
    kappa_init: float,
    xi_init: float,
    censor_threshold: float,
):
    rows = []

    for sp in splits:
        df_train = df_model.loc[sp["train_idx"]].copy()
        df_valid = df_model.loc[sp["valid_idx"]].copy()

        y_train = df_train["Y_obs"].to_numpy(float)
        y_valid = df_valid["Y_obs"].to_numpy(float)

        x_train = df_train[covariate_col].to_numpy(float)
        x_valid = df_valid[covariate_col].to_numpy(float)

        fit = fit_egpd_regression_model(
            y_train=y_train,
            x_train=x_train,
            model_type=model_type,
            variant=variant,
            xi_fixed=xi_init,
            sigma_init=sigma_init,
            kappa_init=kappa_init,
            censor_threshold=censor_threshold,
        )

        pred = predict_egpd_regression_model(
            fit,
            x_valid,
            xi_fixed=xi_init,
        )

        sigma_v = pred["pred_sigma"]
        kappa_v = pred["pred_kappa"]
        xi_v = pred["pred_xi"]

        metrics = summarize_distribution_metrics(
            y_valid,
            sigma_v,
            kappa_v,
            xi_v,
        )

        pred_all = predict_egpd_regression_model(
            fit,
            df_model[covariate_col].to_numpy(float),
            xi_fixed=xi_init,
        )

        y_all = df_model["Y_obs"].to_numpy(float)

        smad_all = smad_exponential_margins(
            y=y_all,
            sigma=pred_all["pred_sigma"],
            kappa=pred_all["pred_kappa"],
            xi=pred_all["pred_xi"],
            p1=0.95,
        )

        row = {
            "fold": sp["fold"],
            "model_family": model_type,
            "model_type": f"{model_type}_1cov",
            "variant": variant,
            "covariate": covariate_col,
            "train_loss": fit["train_nll"],
            "valid_loss": egpd_left_censored_nll(
                y_valid,
                sigma_v,
                kappa_v,
                xi_v,
                c=censor_threshold,
            ),
            "valid_loss_sum": egpd_left_censored_nll_sum(
                y_valid,
                sigma_v,
                kappa_v,
                xi_v,
                c=censor_threshold,
            ),
            "n_train": len(y_train),
            "n_valid": len(y_valid),
            "success": fit["success"],
            "smad_original": smad_all["smad"],
            "smad_original_m": smad_all["smad_m"],
        }

        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


# %%
# Comparison summary
def summarize_model_comparison(res: pd.DataFrame) -> pd.DataFrame:
    res = res.copy()

    group_cols = ["model_family", "model_type", "variant", "covariate"]

    res["covariate"] = res["covariate"].fillna("none")

    agg_dict = {
        "n_folds": ("fold", "count"),
        "valid_loss_mean": ("valid_loss", "mean"),
        "valid_loss_std": ("valid_loss", "std"),
        "valid_loss_sum_mean": ("valid_loss_sum", "mean"),
        "crps_mean": ("crps_mean", "mean"),
        "crps_sum_mean": ("crps_sum", "mean"),
        "twcrps_paper_sum_mean": ("twcrps_paper_sum", "mean"),
        "twcrps_paper_mean_obs_mean": ("twcrps_paper_mean_obs", "mean"),
        "twcrps_paper_mean_obs_thr_mean": ("twcrps_paper_mean_obs_thr", "mean"),
        "smad_mean": ("smad", "mean"),
        "smad_std": ("smad", "std"),
        "smad_original_mean": ("smad_original", "mean"),
        "smad_original_std": ("smad_original", "std"),
        "err50_mean": ("abs_calib_err_0.50", "mean"),
        "err95_mean": ("abs_calib_err_0.95", "mean"),
        "err99_mean": ("abs_calib_err_0.99", "mean"),
        "mean_abs_err_mean": ("mean_abs_err", "mean"),
    }

    agg_dict = {
        k: v
        for k, v in agg_dict.items()
        if v[0] in res.columns
    }

    summary = (
        res.groupby(group_cols, as_index=False)
        .agg(**agg_dict)
    )

    delta_metric_cols = [
        "valid_loss_mean",
        "valid_loss_sum_mean",
        "crps_mean",
        "crps_sum_mean",
        "twcrps_paper_sum_mean",
        "twcrps_paper_mean_obs_mean",
        "twcrps_paper_mean_obs_thr_mean",
        "smad_mean",
        "smad_original_mean",
        "err50_mean",
        "err95_mean",
        "err99_mean",
        "mean_abs_err_mean",
    ]

    for col in delta_metric_cols:
        if col in summary.columns:
            delta_col = col.replace("_mean", "_delta")
            summary[delta_col] = summary[col] - summary[col].min()

    rank_cols = [
        "valid_loss_mean",
        "crps_mean",
        "twcrps_paper_sum_mean",
        "smad_original_mean",
        "smad_mean",
        "err95_mean",
        "err99_mean",
    ]

    for col in rank_cols:
        if col in summary.columns:
            summary[f"rank_{col.replace('_mean', '')}"] = summary[col].rank(method="average")

    sort_cols = [
        c for c in [
            "valid_loss_mean",
            "twcrps_paper_sum_mean",
            "crps_mean",
            "smad_original_mean",
        ]
        if c in summary.columns
    ]

    summary = summary.sort_values(sort_cols).reset_index(drop=True)

    return summary


# %%
# Load data
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

single_cov_col = "X_p01_dt0h"

if single_cov_col not in df_model.columns:
    raise ValueError(f"{single_cov_col} not found in df_model columns.")

print(f"Using single covariate: {single_cov_col}")
# %%
# Train/valid vs test split
OUT_TEST = os.path.join(DATA_FOLDER, "downscaling/model_comparison_test.csv")
OUT_TEST_PRED = os.path.join(DATA_FOLDER, "downscaling/model_diagnostic_predictions_test.csv")


def make_train_valid_test_split(
    df: pd.DataFrame,
    test_frac: float = 0.10,
    block: str = "30D",
    seed: int = 2026,
):
    """
    Split df into train_valid and test using time blocks.

    The test set is removed before any CV/tuning/model selection.
    The fraction is block-based, therefore the observation fraction is approximate.
    """
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True)

    block_labels = make_time_blocks(d, block=block)
    unique_blocks = np.array(sorted(block_labels.unique()))

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_blocks)

    n_test_blocks = int(np.ceil(test_frac * len(unique_blocks)))
    n_test_blocks = max(1, min(n_test_blocks, len(unique_blocks) - 1))

    test_blocks = set(unique_blocks[:n_test_blocks])
    train_valid_blocks = set(unique_blocks[n_test_blocks:])

    train_valid_idx = d.index[block_labels.isin(train_valid_blocks)].to_numpy()
    test_idx = d.index[block_labels.isin(test_blocks)].to_numpy()

    df_train_valid = d.loc[train_valid_idx].copy()
    df_test = d.loc[test_idx].copy()

    info = {
        "test_frac_requested": test_frac,
        "test_frac_observed": len(df_test) / len(d),
        "n_total": len(d),
        "n_train_valid": len(df_train_valid),
        "n_test": len(df_test),
        "n_blocks_total": len(unique_blocks),
        "n_blocks_train_valid": len(train_valid_blocks),
        "n_blocks_test": len(test_blocks),
        "test_blocks": sorted(test_blocks),
        "train_valid_blocks": sorted(train_valid_blocks),
    }

    return df_train_valid, df_test, info


df_train_valid, df_test, test_split_info = make_train_valid_test_split(
    df=df_model,
    test_frac=0.10,
    block="30D",
    seed=2026,
)

print("\n=== Train/valid vs test split ===")
for k, v in test_split_info.items():
    if k not in {"test_blocks", "train_valid_blocks"}:
        print(f"{k}: {v}")

print("\nTrain/valid period:")
print(df_train_valid["time"].min(), "->", df_train_valid["time"].max())

print("\nTest period:")
print(df_test["time"].min(), "->", df_test["time"].max())


# %%
# CV splits on train_valid

cv_splits = make_blocked_cv_splits(
    df=df_train_valid,
    n_splits=3,
    block="30D",
    seed=1,
)

print("\n=== CV folds on train_valid only ===")
for sp in cv_splits:
    print(
        f"fold={sp['fold']} | "
        f"n_train={len(sp['train_idx'])} | "
        f"n_valid={len(sp['valid_idx'])} | "
        f"n_valid_blocks={len(sp['valid_blocks'])}"
    )


# %%
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

filename = os.path.join(IM_FOLDER, "cv_splits_with_heldout_test.png")
fig.savefig(filename, dpi=300, bbox_inches="tight")
print("Saved:", filename)


# %%
# NN covariate sets and tuning grid
TUNING_PRESET = "large"

x_sets = make_covariate_sets(
    x_cols27=x_cols27,
    x_cols_dt0h=x_cols_dt0h,
    x_cols_all=x_cols_all,
)

print("\nAvailable NN covariate sets:")
for name, cols in x_sets.items():
    print(f"{name:20s}: {len(cols)} variables")


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


# %%
# Run stationary / GLM / GAM models on CV folds of train_valid only
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

#%%

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


# %%
# NN tuning on train_valid only
print("\nTuning NN once on outer fold 0 of train_valid only")

tuning_df, best_params = tune_nn_on_outer_train(
    df_model=df_train_valid,
    outer_split=cv_splits[0],
    x_sets=x_sets,
    param_grid=nn_param_grid,
    seed=1,
    device=None,
)

print("\nBest NN params found on fold 0:")
print(best_params)

print("\nTop tuning results:")
print(tuning_df.head(30).to_string(index=False))

tuning_df.to_csv(OUT_TUNING, index=False)

print("\nSaved fast tuning history to:")
print(OUT_TUNING)


# %%
# Re-ranking top NN configs on train_valid only
print("\nRe-ranking top NN configs with validation loss / paper-like twCRPS / sMAD")

rerank_df, best_params = rerank_top_nn_configs(
    df_model=df_train_valid,
    outer_split=cv_splits[0],
    x_sets=x_sets,
    tuning_df=tuning_df,
    top_k=10,
    seed=1,
    device=None,
)

print("\nBest NN params after re-ranking:")
print(best_params)

print("\nTop re-ranked results:")
print(rerank_df.head(20).to_string(index=False))

rerank_df.to_csv(OUT_RERANK, index=False)
pd.DataFrame([best_params]).to_csv(OUT_BEST_PARAMS, index=False)

print("\nSaved re-ranking results to:")
print(OUT_RERANK)

print("\nSaved final best NN params to:")
print(OUT_BEST_PARAMS)


# %%
# Run fixed tuned NN on all train_valid CV folds
print("\nRunning fixed tuned NN on all train_valid folds")

best_params_final = best_params.copy()
best_params_final["n_ep"] = 300

res_nn_tuned = evaluate_fixed_nn_model(
    df_model=df_train_valid,
    splits=cv_splits,
    x_sets=x_sets,
    best_params=best_params_final,
    seed=1,
    device=None,
)

all_results.append(res_nn_tuned)


# %%
# Save and summarize CV results
comparison_res = pd.concat(
    all_results,
    ignore_index=True,
    sort=False,
)

comparison_res.to_csv(
    OUT_COMPARISON,
    index=False,
)

print("\nSaved train_valid CV fold-level results to:")
print(OUT_COMPARISON)

summary = summarize_model_comparison(comparison_res)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)

print("\nSaved train_valid CV comparison summary to:")
print(OUT_SUMMARY)

print("\n=== TRAIN_VALID CV COMPARISON SUMMARY ===")
print(summary.to_string(index=False))

print("\n=== TOP MODELS ON TRAIN_VALID CV ===")
print(summary.head(15).to_string(index=False))


# %%
# Delta-to-best summary on train_valid CV
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

print("\nSaved train_valid CV delta-to-best summary to:")
print(OUT_SUMMARY_DELTA)

print("\n=== TRAIN_VALID CV SUMMARY: DIFFERENCE TO BEST MODEL ===")
print(summary_delta.to_string(index=False))


# %%
# Quick tuning diagnostics
print("\n=== Best NN valid loss by covariate set and init source ===")
print(
    tuning_df
    .groupby(["x_set_name", "init_source"])["valid_loss"]
    .min()
    .sort_values()
    .to_string()
)

print("\n=== Top 30 NN tuning configurations ===")

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


# %%
# Final test evaluation
def make_test_metric_row(
    y_test,
    sigma_t,
    kappa_t,
    xi_t,
    model_family: str,
    model_type: str,
    variant: str,
    covariate: str,
    train_loss: float,
    test_loss: float,
    test_loss_sum: float,
    extra: Optional[dict] = None,
):
    metrics = summarize_distribution_metrics(
        y_test,
        sigma_t,
        kappa_t,
        xi_t,
    )

    row = {
        "model_family": model_family,
        "model_type": model_type,
        "variant": variant,
        "covariate": covariate,
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),
        "test_loss_sum": float(test_loss_sum),
        "n_train_valid": len(df_train_valid),
        "n_test": len(df_test),
    }

    row.update(metrics)

    if extra is not None:
        row.update(extra)

    return row


def make_prediction_df(
    df_test: pd.DataFrame,
    model_label: str,
    sigma_t,
    kappa_t,
    xi_t,
):
    pred = df_test[["time", "station", "Y_obs"]].copy()
    pred["fold"] = "test"
    pred["model"] = model_label
    pred["sigma"] = np.asarray(sigma_t, dtype=float)
    pred["kappa"] = np.asarray(kappa_t, dtype=float)
    pred["xi"] = np.asarray(xi_t, dtype=float)
    return pred


def fit_predict_stationary_test(
    df_train_valid: pd.DataFrame,
    df_test: pd.DataFrame,
):
    y_train = df_train_valid["Y_obs"].to_numpy(float)
    y_test = df_test["Y_obs"].to_numpy(float)

    fit = fit_egpd_stationary_direct(
        y_train=y_train,
        y_valid=y_test,
        sigma_init=SIGMA_INIT,
        kappa_init=KAPPA_INIT,
        xi_init=XI_INIT,
        fix_xi=True,
        censor_threshold=CENSOR_THRESHOLD,
    )

    sigma_t = np.full(len(y_test), fit["sigma_hat"], dtype=float)
    kappa_t = np.full(len(y_test), fit["kappa_hat"], dtype=float)
    xi_t = np.full(len(y_test), fit["xi_hat"], dtype=float)

    test_loss = egpd_left_censored_nll(
        y_test,
        sigma_t,
        kappa_t,
        xi_t,
        c=CENSOR_THRESHOLD,
    )

    test_loss_sum = egpd_left_censored_nll_sum(
        y_test,
        sigma_t,
        kappa_t,
        xi_t,
        c=CENSOR_THRESHOLD,
    )

    row = make_test_metric_row(
        y_test=y_test,
        sigma_t=sigma_t,
        kappa_t=kappa_t,
        xi_t=xi_t,
        model_family="stationary",
        model_type="stationary",
        variant="stationary",
        covariate="none",
        train_loss=fit["train_nll"],
        test_loss=test_loss,
        test_loss_sum=test_loss_sum,
        extra={
            "sigma_hat": fit["sigma_hat"],
            "kappa_hat": fit["kappa_hat"],
            "xi_hat": fit["xi_hat"],
            "success": fit["success"],
        },
    )

    pred = make_prediction_df(
        df_test=df_test,
        model_label="Simple fit",
        sigma_t=sigma_t,
        kappa_t=kappa_t,
        xi_t=xi_t,
    )

    return row, pred


def fit_predict_regression_test(
    df_train_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    model_type: str,
    variant: str,
    covariate_col: str,
):
    y_train = df_train_valid["Y_obs"].to_numpy(float)
    y_test = df_test["Y_obs"].to_numpy(float)

    x_train = df_train_valid[covariate_col].to_numpy(float)
    x_test = df_test[covariate_col].to_numpy(float)

    fit = fit_egpd_regression_model(
        y_train=y_train,
        x_train=x_train,
        model_type=model_type,
        variant=variant,
        xi_fixed=XI_INIT,
        sigma_init=SIGMA_INIT,
        kappa_init=KAPPA_INIT,
        censor_threshold=CENSOR_THRESHOLD,
    )

    pred_params = predict_egpd_regression_model(
        fit,
        x_test,
        xi_fixed=XI_INIT,
    )

    sigma_t = pred_params["pred_sigma"]
    kappa_t = pred_params["pred_kappa"]
    xi_t = pred_params["pred_xi"]

    test_loss = egpd_left_censored_nll(
        y_test,
        sigma_t,
        kappa_t,
        xi_t,
        c=CENSOR_THRESHOLD,
    )

    test_loss_sum = egpd_left_censored_nll_sum(
        y_test,
        sigma_t,
        kappa_t,
        xi_t,
        c=CENSOR_THRESHOLD,
    )

    row = make_test_metric_row(
        y_test=y_test,
        sigma_t=sigma_t,
        kappa_t=kappa_t,
        xi_t=xi_t,
        model_family=model_type,
        model_type=f"{model_type}_1cov",
        variant=variant,
        covariate=covariate_col,
        train_loss=fit["train_nll"],
        test_loss=test_loss,
        test_loss_sum=test_loss_sum,
        extra={"success": fit["success"]},
    )

    pred = make_prediction_df(
        df_test=df_test,
        model_label=f"{model_type.upper()} {variant}",
        sigma_t=sigma_t,
        kappa_t=kappa_t,
        xi_t=xi_t,
    )

    return row, pred


def fit_predict_nn_test(
    df_train_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    x_sets: dict[str, list[str]],
    best_params: dict,
    seed: int = 123,
    device: Optional[str] = None,
):
    """
    Final NN fit:
    - uses train_valid only;
    - creates an internal validation split inside train_valid for early stopping;
    - predicts on held-out test only after model selection.
    """
    x_set_name = best_params["x_set_name"]
    x_cols = x_sets[x_set_name]

    final_split = make_single_split_from_train(
        df_train=df_train_valid,
        train_frac=0.90,
        block="30D",
        seed=seed,
    )

    cfg = Config(
        name=f"nn_final_{x_set_name}",
        widths=parse_widths(best_params["widths"]),
    )

    fit, meta = run_one_nn_variant(
        df_model=df_train_valid,
        split=final_split,
        x_cols=x_cols,
        cfg=cfg,
        variant=best_params["variant"],
        sigma_init=float(best_params["sigma_init"]),
        kappa_init=float(best_params["kappa_init"]),
        xi_init=float(best_params["xi_init"]),
        lr=float(best_params["lr"]),
        n_ep=int(best_params["n_ep"]),
        batch_size=int(best_params["batch_size"]),
        seed=seed,
        device=device,
        censor_threshold=float(best_params["censor_threshold"]),
        early_stopping=True,
        patience=25,
        warmup_epochs=20,
        weight_decay=float(best_params.get("weight_decay", 0.0)),
    )

    y_test = df_test["Y_obs"].to_numpy(float)

    sigma_t, kappa_t, xi_t = predict_params_on_df_variant(
        df_test,
        fit,
        meta,
    )

    test_loss = egpd_left_censored_nll(
        y_test,
        sigma_t,
        kappa_t,
        xi_t,
        c=float(best_params["censor_threshold"]),
    )

    test_loss_sum = egpd_left_censored_nll_sum(
        y_test,
        sigma_t,
        kappa_t,
        xi_t,
        c=float(best_params["censor_threshold"]),
    )

    row = make_test_metric_row(
        y_test=y_test,
        sigma_t=sigma_t,
        kappa_t=kappa_t,
        xi_t=xi_t,
        model_family="nn",
        model_type="nn_tuned_final",
        variant=best_params["variant"],
        covariate=x_set_name,
        train_loss=float(fit["train_nll"]),
        test_loss=test_loss,
        test_loss_sum=test_loss_sum,
        extra={
            "best_x_set_name": x_set_name,
            "best_n_covariates": len(x_cols),
            "best_widths": str(best_params["widths"]),
            "best_lr": float(best_params["lr"]),
            "best_weight_decay": float(best_params.get("weight_decay", 0.0)),
            "best_batch_size": int(best_params["batch_size"]),
            "best_n_ep": int(best_params["n_ep"]),
            "best_sigma_init": float(best_params["sigma_init"]),
            "best_kappa_init": float(best_params["kappa_init"]),
            "best_xi_init": float(best_params["xi_init"]),
            "best_censor_threshold": float(best_params["censor_threshold"]),
            "best_init_source": best_params.get("init_source", "unknown"),
            "stopped_epoch": int(fit.get("stopped_epoch", np.nan)),
            "internal_final_train_n": len(final_split["train_idx"]),
            "internal_final_valid_n": len(final_split["valid_idx"]),
        },
    )

    pred = make_prediction_df(
        df_test=df_test,
        model_label=f"NN {x_set_name}",
        sigma_t=sigma_t,
        kappa_t=kappa_t,
        xi_t=xi_t,
    )

    return row, pred


# %%
# Final evaluation on test
test_rows = []
test_preds = []

# Stationary final
row, pred = fit_predict_stationary_test(
    df_train_valid=df_train_valid,
    df_test=df_test,
)
test_rows.append(row)
test_preds.append(pred)

# GLM/GAM final models
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

# Tuned NN final model
row, pred = fit_predict_nn_test(
    df_train_valid=df_train_valid,
    df_test=df_test,
    x_sets=x_sets,
    best_params=best_params_final,
    seed=123,
    device=None,
)
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
# Functions for diagnostic predictions and figures on test set
# %%
# Functions for diagnostic predictions and figures on test set

from typing import Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Diagnostic prediction quantities for EGPD
# =========================================================
def add_prediction_quantities(
    df_pred: pd.DataFrame,
    y_col: str = "Y_obs",
    quantiles: Sequence[float] = (0.5, 0.75, 0.9, 0.95, 0.99),
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Add diagnostic quantities for EGPD predictive distributions.

    Expected columns:
        y_col, sigma, kappa, xi

    Added columns:
        q50_pred, q75_pred, q90_pred, q95_pred, q99_pred
        mean_pred
        pit
        cdf_obs
        exp_residual
        ae_q50
        se_q50
    """
    out = df_pred.copy()

    required = [y_col, "sigma", "kappa", "xi"]
    missing = [c for c in required if c not in out.columns]

    if missing:
        raise ValueError(f"Missing columns in df_pred: {missing}")

    y = out[y_col].to_numpy(dtype=float)
    sigma = out["sigma"].to_numpy(dtype=float)
    kappa = out["kappa"].to_numpy(dtype=float)
    xi = out["xi"].to_numpy(dtype=float)

    for q in quantiles:
        q_int = int(round(100 * q))
        out[f"q{q_int}_pred"] = qeGPD(
            prob=np.full(len(out), q),
            sigma=sigma,
            kappa=kappa,
            xi=xi,
            eps=eps,
        )

    out["mean_pred"] = egpd_mean(
        sigma=sigma,
        kappa=kappa,
        xi=xi,
        eps=eps,
    )

    pit = egpd_cdf(
        y=y,
        sigma=sigma,
        kappa=kappa,
        xi=xi,
        eps=eps,
    )

    pit = np.clip(pit, eps, 1.0 - eps)

    out["pit"] = pit
    out["cdf_obs"] = pit
    out["exp_residual"] = -np.log(1.0 - pit)

    if "q50_pred" in out.columns:
        out["ae_q50"] = np.abs(y - out["q50_pred"].to_numpy(dtype=float))
        out["se_q50"] = (y - out["q50_pred"].to_numpy(dtype=float)) ** 2

    return out


# =========================================================
# Plot helpers
# =========================================================
def _get_models_in_order(
    df: pd.DataFrame,
    model_col: str = "model",
    model_order: Optional[Sequence[str]] = None,
) -> list:
    """
    Return model names in the requested order, keeping only models present in df.
    """
    if model_col not in df.columns:
        raise ValueError(f"Column '{model_col}' not found in dataframe.")

    present = list(pd.unique(df[model_col]))

    if model_order is None:
        return present

    return [m for m in model_order if m in set(present)]


def _finalize_and_save(
    fig,
    save_name: Optional[str] = None,
    save_dir: Optional[str] = None,
    dpi: int = 300,
):
    """
    Show the figure and optionally save it as PNG and PDF.

    If save_name is None, the figure is only shown.
    """
    fig.tight_layout()

    if save_name is not None:
        if save_dir is None:
            save_dir = DIAG_DIR

        os.makedirs(save_dir, exist_ok=True)

        png_path = os.path.join(save_dir, f"{save_name}.png")
        pdf_path = os.path.join(save_dir, f"{save_name}.pdf")

        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")

        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")

    plt.show()


# =========================================================
# 1) Exponential QQ plot
# =========================================================
def plot_exponential_qq(
    df: pd.DataFrame,
    model_order: Optional[Sequence[str]] = None,
    model_col: str = "model",
    exp_col: str = "exp_residual",
    p_low: float = 0.50,
    p_high: float = 0.995,
    n_probs: int = 150,
    figsize: Tuple[float, float] = (7, 7),
    save_name: Optional[str] = None,
):
    """
    QQ plot of transformed PIT residuals against Exp(1).

    If F(Y) is uniform, then:
        Z = -log(1 - F(Y))
    should follow Exp(1).
    """
    if exp_col not in df.columns:
        raise ValueError(f"Column '{exp_col}' not found in dataframe.")

    models = _get_models_in_order(df, model_col=model_col, model_order=model_order)

    probs = np.linspace(p_low, p_high, n_probs)
    theo = -np.log(1.0 - probs)

    fig, ax = plt.subplots(figsize=figsize)

    for model in models:
        vals = (
            df.loc[df[model_col] == model, exp_col]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(vals) < 5:
            continue

        emp = np.quantile(vals, probs)

        ax.plot(
            theo,
            emp,
            label=model,
        )

    lim_min = float(np.nanmin(theo))
    lim_max = float(np.nanmax(theo))

    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        linestyle="--",
    )

    ax.set_xlabel("Theoretical Exp(1) quantiles")
    ax.set_ylabel("Empirical transformed quantiles")
    ax.set_title("Exponential QQ plot")
    ax.legend()
    ax.grid(alpha=0.3)

    _finalize_and_save(fig, save_name=save_name)


# =========================================================
# 2) PIT histograms
# =========================================================
def plot_pit_histograms(
    df: pd.DataFrame,
    model_order: Optional[Sequence[str]] = None,
    model_col: str = "model",
    pit_col: str = "pit",
    n_bins: int = 20,
    figsize_per_panel: Tuple[float, float] = (4.5, 3.5),
    save_name: Optional[str] = None,
):
    """
    PIT histograms, one panel per model.

    A well-calibrated continuous predictive distribution should give
    approximately uniform PIT values.
    """
    if pit_col not in df.columns:
        raise ValueError(f"Column '{pit_col}' not found in dataframe.")

    models = _get_models_in_order(df, model_col=model_col, model_order=model_order)

    if len(models) == 0:
        raise ValueError("No models available for plotting.")

    n = len(models)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]),
        squeeze=False,
    )

    for ax, model in zip(axes.ravel(), models):
        vals = (
            df.loc[df[model_col] == model, pit_col]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
        )

        vals = vals[(vals >= 0.0) & (vals <= 1.0)]

        ax.hist(
            vals,
            bins=n_bins,
            range=(0.0, 1.0),
            density=True,
        )

        ax.axhline(1.0, linestyle="--")
        ax.set_title(model)
        ax.set_xlabel("PIT")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)

    fig.suptitle("PIT histograms", y=1.03)

    _finalize_and_save(fig, save_name=save_name)


# =========================================================
# 3) Predicted vs observed
# =========================================================
def plot_predicted_vs_observed(
    df: pd.DataFrame,
    model_order: Optional[Sequence[str]] = None,
    pred_col: str = "q50_pred",
    y_col: str = "Y_obs",
    model_col: str = "model",
    figsize_per_panel: Tuple[float, float] = (4.5, 4.0),
    alpha: float = 0.35,
    save_name: Optional[str] = None,
):
    """
    Scatter plot of predicted quantity versus observed value.
    """
    required = [y_col, pred_col, model_col]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    models = _get_models_in_order(df, model_col=model_col, model_order=model_order)

    if len(models) == 0:
        raise ValueError("No models available for plotting.")

    n = len(models)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]),
        squeeze=False,
    )

    for ax, model in zip(axes.ravel(), models):
        d = df.loc[df[model_col] == model, [y_col, pred_col]].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna()

        if len(d) == 0:
            ax.set_title(model)
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        x = d[pred_col].to_numpy(dtype=float)
        y = d[y_col].to_numpy(dtype=float)

        ax.scatter(
            x,
            y,
            alpha=alpha,
            s=12,
        )

        vmax = np.nanmax(
            np.r_[
                x[np.isfinite(x)],
                y[np.isfinite(y)],
            ]
        )

        vmax = max(float(vmax), 1e-8)

        ax.plot(
            [0.0, vmax],
            [0.0, vmax],
            linestyle="--",
        )

        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.set_title(model)
        ax.set_xlabel(pred_col)
        ax.set_ylabel(y_col)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Predicted vs observed: {pred_col}", y=1.03)

    _finalize_and_save(fig, save_name=save_name)


# =========================================================
# 4) Quantile calibration
# =========================================================
def plot_quantile_calibration(
    df: pd.DataFrame,
    model_order: Optional[Sequence[str]] = None,
    model_col: str = "model",
    y_col: str = "Y_obs",
    q_levels: Sequence[float] = (0.5, 0.75, 0.9, 0.95, 0.99),
    figsize: Tuple[float, float] = (7, 6),
    save_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Quantile calibration plot.

    For each nominal quantile q, computes:
        empirical coverage = mean(Y_obs <= q_pred)

    Returns the calibration dataframe.
    """
    required = [y_col, model_col]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    models = _get_models_in_order(df, model_col=model_col, model_order=model_order)

    rows = []

    for model in models:
        d = df.loc[df[model_col] == model].copy()

        for q in q_levels:
            q_int = int(round(100 * q))
            q_col = f"q{q_int}_pred"

            if q_col not in d.columns:
                rows.append({
                    "model": model,
                    "q_level": q,
                    "empirical_coverage": np.nan,
                    "abs_calibration_error": np.nan,
                    "n": 0,
                })
                continue

            dd = d[[y_col, q_col]].replace([np.inf, -np.inf], np.nan).dropna()

            if len(dd) == 0:
                emp = np.nan
                abs_err = np.nan
            else:
                emp = float(
                    np.mean(
                        dd[y_col].to_numpy(dtype=float)
                        <= dd[q_col].to_numpy(dtype=float)
                    )
                )
                abs_err = float(abs(emp - q))

            rows.append({
                "model": model,
                "q_level": float(q),
                "empirical_coverage": emp,
                "abs_calibration_error": abs_err,
                "n": len(dd),
            })

    calib_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=figsize)

    for model in models:
        dd = calib_df.loc[calib_df["model"] == model]

        if len(dd) == 0:
            continue

        ax.plot(
            dd["q_level"].to_numpy(dtype=float),
            dd["empirical_coverage"].to_numpy(dtype=float),
            marker="o",
            label=model,
        )

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Nominal quantile level")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Quantile calibration")
    ax.legend()
    ax.grid(alpha=0.3)

    _finalize_and_save(fig, save_name=save_name)

    return calib_df


# =========================================================
# 5) Tail exceedance calibration
# =========================================================
def plot_tail_exceedance_calibration(
    df: pd.DataFrame,
    model_order: Optional[Sequence[str]] = None,
    thresholds: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0),
    model_col: str = "model",
    y_col: str = "Y_obs",
    figsize: Tuple[float, float] = (7, 6),
    save_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare empirical exceedance probabilities P(Y > t)
    to mean predicted EGPD exceedance probabilities.

    Returns a dataframe with:
        model, threshold, empirical_exceedance, predicted_exceedance, abs_error, n
    """
    required = [y_col, model_col, "sigma", "kappa", "xi"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    models = _get_models_in_order(df, model_col=model_col, model_order=model_order)

    rows = []

    for model in models:
        d = df.loc[df[model_col] == model].copy()

        d = (
            d[[y_col, "sigma", "kappa", "xi"]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(d) == 0:
            continue

        y = d[y_col].to_numpy(dtype=float)
        sigma = d["sigma"].to_numpy(dtype=float)
        kappa = d["kappa"].to_numpy(dtype=float)
        xi = d["xi"].to_numpy(dtype=float)

        for t in thresholds:
            empirical_exceedance = float(np.mean(y > t))

            predicted_exceedance = float(
                np.mean(
                    1.0 - egpd_cdf(
                        y=np.full(len(d), t),
                        sigma=sigma,
                        kappa=kappa,
                        xi=xi,
                    )
                )
            )

            rows.append({
                "model": model,
                "threshold": float(t),
                "empirical_exceedance": empirical_exceedance,
                "predicted_exceedance": predicted_exceedance,
                "abs_error": abs(empirical_exceedance - predicted_exceedance),
                "n": len(d),
            })

    exc_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=figsize)

    for model in models:
        dd = exc_df.loc[exc_df["model"] == model]

        if len(dd) == 0:
            continue

        ax.plot(
            dd["predicted_exceedance"].to_numpy(dtype=float),
            dd["empirical_exceedance"].to_numpy(dtype=float),
            marker="o",
            label=model,
        )

        for _, row in dd.iterrows():
            ax.annotate(
                f"{row['threshold']}",
                (
                    row["predicted_exceedance"],
                    row["empirical_exceedance"],
                ),
                fontsize=8,
                xytext=(3, 3),
                textcoords="offset points",
            )

    if len(exc_df) > 0:
        maxv = float(
            np.nanmax(
                np.r_[
                    exc_df["predicted_exceedance"].to_numpy(dtype=float),
                    exc_df["empirical_exceedance"].to_numpy(dtype=float),
                ]
            )
        )
    else:
        maxv = 1e-6

    maxv = max(maxv, 1e-6)

    ax.plot(
        [0.0, maxv],
        [0.0, maxv],
        linestyle="--",
    )

    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Mean predicted exceedance probability")
    ax.set_ylabel("Empirical exceedance probability")
    ax.set_title("Tail exceedance calibration")
    ax.legend()
    ax.grid(alpha=0.3)

    _finalize_and_save(fig, save_name=save_name)

    return exc_df


# =========================================================
# 6) Optional: parameter distributions
# =========================================================
def plot_parameter_distributions(
    df: pd.DataFrame,
    model_order: Optional[Sequence[str]] = None,
    model_col: str = "model",
    param_cols: Sequence[str] = ("sigma", "kappa", "xi"),
    n_bins: int = 30,
    figsize_per_panel: Tuple[float, float] = (4.5, 3.5),
    save_name: Optional[str] = None,
):
    """
    Plot predicted parameter distributions by model.
    """
    required = [model_col, *param_cols]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    models = _get_models_in_order(df, model_col=model_col, model_order=model_order)
    n_params = len(param_cols)

    fig, axes = plt.subplots(
        n_params,
        1,
        figsize=(figsize_per_panel[0], figsize_per_panel[1] * n_params),
        squeeze=False,
    )

    for ax, param in zip(axes.ravel(), param_cols):
        for model in models:
            vals = (
                df.loc[df[model_col] == model, param]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .to_numpy(dtype=float)
            )

            if len(vals) == 0:
                continue

            ax.hist(
                vals,
                bins=n_bins,
                alpha=0.35,
                density=True,
                label=model,
            )

        ax.set_title(f"Distribution of {param}")
        ax.set_xlabel(param)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        ax.legend()

    _finalize_and_save(fig, save_name=save_name)


# =========================================================
# 7) Optional: diagnostic summary table
# =========================================================
def summarize_prediction_diagnostics(
    df: pd.DataFrame,
    model_order: Optional[Sequence[str]] = None,
    model_col: str = "model",
    y_col: str = "Y_obs",
) -> pd.DataFrame:
    """
    Summarize basic diagnostics from pred_test_all.

    Requires columns:
        model, Y_obs, pit, exp_residual, q50_pred, q95_pred, mean_pred
    """
    required = [model_col, y_col, "pit", "exp_residual"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    models = _get_models_in_order(df, model_col=model_col, model_order=model_order)

    rows = []

    for model in models:
        d = df.loc[df[model_col] == model].copy()
        d = d.replace([np.inf, -np.inf], np.nan)

        y = d[y_col].to_numpy(dtype=float)

        row = {
            "model": model,
            "n": int(len(d)),
            "obs_mean": float(np.nanmean(y)),
            "pit_mean": float(np.nanmean(d["pit"])),
            "pit_var": float(np.nanvar(d["pit"], ddof=1)),
            "pit_mean_err": float(abs(np.nanmean(d["pit"]) - 0.5)),
            "pit_var_err": float(abs(np.nanvar(d["pit"], ddof=1) - 1.0 / 12.0)),
            "exp_residual_mean": float(np.nanmean(d["exp_residual"])),
            "exp_residual_q95": float(np.nanquantile(d["exp_residual"], 0.95)),
        }

        if "q50_pred" in d.columns:
            row["mae_q50"] = float(
                np.nanmean(
                    np.abs(
                        d[y_col].to_numpy(dtype=float)
                        - d["q50_pred"].to_numpy(dtype=float)
                    )
                )
            )

        if "mean_pred" in d.columns:
            row["mean_pred_avg"] = float(np.nanmean(d["mean_pred"]))
            row["mean_abs_err"] = float(
                abs(row["mean_pred_avg"] - row["obs_mean"])
            )

        if "q95_pred" in d.columns:
            row["cover_q95"] = float(
                np.nanmean(
                    d[y_col].to_numpy(dtype=float)
                    <= d["q95_pred"].to_numpy(dtype=float)
                )
            )
            row["cover_q95_abs_err"] = float(abs(row["cover_q95"] - 0.95))

        if "q99_pred" in d.columns:
            row["cover_q99"] = float(
                np.nanmean(
                    d[y_col].to_numpy(dtype=float)
                    <= d["q99_pred"].to_numpy(dtype=float)
                )
            )
            row["cover_q99_abs_err"] = float(abs(row["cover_q99"] - 0.99))

        rows.append(row)

    return pd.DataFrame(rows)


# %%
# Diagnostic predictions
pred_test_all = pd.concat(
    test_preds,
    ignore_index=True,
    sort=False,
)

pred_test_all = add_prediction_quantities(pred_test_all)

pred_test_all.to_csv(OUT_TEST_PRED, index=False)


# %%
# Diagnostic figures
nn_model_label = f"NN {best_params_final['x_set_name']}"

model_order = [
    "Simple fit",
    "GLM both",
    "GAM both",
    nn_model_label,
]

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
# %%
def compare_train_test_distribution(
    df_train_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    y_col: str = "Y_obs",
):
    probs = [0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995]

    rows = []

    for name, d in [
        ("train_valid", df_train_valid),
        ("test", df_test),
    ]:
        y = d[y_col].to_numpy(float)
        y = y[np.isfinite(y)]

        row = {
            "set": name,
            "n": len(y),
            "mean": np.mean(y),
            "std": np.std(y, ddof=1),
            "max": np.max(y),
        }

        for p in probs:
            row[f"q{int(1000*p):03d}"] = np.quantile(y, p)

        for thr in [0.5, 1, 2, 5, 10, 20]:
            row[f"p_gt_{thr}"] = np.mean(y > thr)

        rows.append(row)

    return pd.DataFrame(rows)


dist_check = compare_train_test_distribution(
    df_train_valid,
    df_test,
    y_col="Y_obs",
)

print(dist_check.to_string(index=False))
# %%
def pit_by_observed_bins(
    pred_test_all: pd.DataFrame,
    y_col: str = "Y_obs",
    model_col: str = "model",
):
    out = pred_test_all.copy()

    out["y_bin"] = pd.cut(
        out[y_col],
        bins=[0, 0.5, 1, 2, 5, 10, np.inf],
        right=True,
        include_lowest=True,
    )

    summary = (
        out.groupby([model_col, "y_bin"], observed=True)
        .agg(
            n=("pit", "size"),
            pit_mean=("pit", "mean"),
            pit_q25=("pit", lambda x: np.quantile(x.dropna(), 0.25)),
            pit_q50=("pit", lambda x: np.quantile(x.dropna(), 0.50)),
            pit_q75=("pit", lambda x: np.quantile(x.dropna(), 0.75)),
            y_mean=(y_col, "mean"),
        )
        .reset_index()
    )

    return summary


pit_bins = pit_by_observed_bins(pred_test_all)
print(pit_bins.to_string(index=False))
# %%
def plot_train_test_distribution_shift(
    df_train_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    y_col: str = "Y_obs",
    save_name: Optional[str] = "train_test_distribution_shift",
):
    probs = np.linspace(0.5, 0.995, 100)

    y_train = df_train_valid[y_col].to_numpy(float)
    y_test = df_test[y_col].to_numpy(float)

    y_train = y_train[np.isfinite(y_train)]
    y_test = y_test[np.isfinite(y_test)]

    q_train = np.quantile(y_train, probs)
    q_test = np.quantile(y_test, probs)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(probs, q_train, label="train_valid")
    ax.plot(probs, q_test, label="test")

    ax.set_xlabel("Quantile level")
    ax.set_ylabel(y_col)
    ax.set_title("Train-valid vs test distribution shift")
    ax.grid(alpha=0.3)
    ax.legend()

    _finalize_and_save(fig, save_name=save_name)

    return pd.DataFrame({
        "prob": probs,
        "q_train_valid": q_train,
        "q_test": q_test,
        "ratio_test_train": q_test / np.maximum(q_train, 1e-12),
        "diff_test_train": q_test - q_train,
    })
# %%
dist_shift = plot_train_test_distribution_shift(
    df_train_valid,
    df_test,
    y_col="Y_obs",
)
# %%
def repeated_test_distribution_checks(
    df: pd.DataFrame,
    n_repeats: int = 20,
    test_frac: float = 0.10,
    block: str = "30D",
    base_seed: int = 2026,
    y_col: str = "Y_obs",
):
    rows = []

    for r in range(n_repeats):
        df_train_valid_r, df_test_r, info_r = make_train_valid_test_split(
            df=df,
            test_frac=test_frac,
            block=block,
            seed=base_seed + r,
        )

        for set_name, d in [
            ("train_valid", df_train_valid_r),
            ("test", df_test_r),
        ]:
            y = d[y_col].to_numpy(float)
            y = y[np.isfinite(y)]

            rows.append({
                "repeat": r,
                "set": set_name,
                "n": len(y),
                "mean": np.mean(y),
                "q90": np.quantile(y, 0.90),
                "q95": np.quantile(y, 0.95),
                "q99": np.quantile(y, 0.99),
                "p_gt_0.5": np.mean(y > 0.5),
                "p_gt_1": np.mean(y > 1.0),
                "p_gt_2": np.mean(y > 2.0),
                "p_gt_5": np.mean(y > 5.0),
                "test_frac_observed": info_r["test_frac_observed"],
            })

    return pd.DataFrame(rows)
# %%
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
# %%
def plot_exponential_qq_single_model(
    df: pd.DataFrame,
    model_name: str,
    model_col: str = "model",
    exp_col: str = "exp_residual",
    figsize: Tuple[float, float] = (7, 6),
    save_name: Optional[str] = None,
):
    """
    Exponential-margin QQ plot for one model.

    Uses all sorted transformed PIT residuals:
        z = -log(1 - F(Y))

    If the predictive distribution is calibrated, z should follow Exp(1).
    """
    if model_col not in df.columns:
        raise ValueError(f"Column '{model_col}' not found.")
    if exp_col not in df.columns:
        raise ValueError(f"Column '{exp_col}' not found.")

    z = (
        df.loc[df[model_col] == model_name, exp_col]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )

    z = np.sort(z)

    n = len(z)

    if n < 5:
        raise ValueError(f"Not enough data for model {model_name}: n={n}")

    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = -np.log(1.0 - probs)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        theo,
        z,
        s=12,
        alpha=0.65,
    )

    lim_max = float(np.nanmax(np.r_[theo, z]))
    lim_max = max(lim_max, 1e-8)

    ax.plot(
        [0.0, lim_max],
        [0.0, lim_max],
        linestyle="--",
    )

    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Theoretical exponential quantiles")
    ax.set_ylabel("Empirical transformed quantiles")
    ax.set_title(f"Exponential-margin Q-Q plot: {model_name}")
    ax.grid(alpha=0.3)

    _finalize_and_save(fig, save_name=save_name)

#%%
for model_name in model_order:
    plot_exponential_qq_single_model(
        pred_test_all,
        model_name=model_name,
        save_name=f"qq_exponential_{safe_model_name(model_name)}",
    )