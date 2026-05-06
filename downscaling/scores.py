# Scores: CRPS / twCRPS / sMAD
from typing import Optional

import numpy as np
import pandas as pd

from downscaling.egpd import (
    compute_pit,
    egpd_cdf,
    predicted_mean_mc,
    qegpd,
)

qeGPD = qegpd

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
        q_pred = float(np.median(qegpd(p, sigma, kappa, xi)))

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
    n_train_valid: int | None = None,
    n_test: int | None = None,
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
        "n_train_valid": n_train_valid,
        "n_test": len(y_test) if n_test is None else n_test,
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
