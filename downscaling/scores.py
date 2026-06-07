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

from typing import Sequence
from downscaling.config import (
    KAPPA_LIMIT, REFERENCE_MODEL, MODEL_ORDER, MODEL_ORDER_NO_REF, STATION_COL_CANDIDATES
)

from downscaling.diagnostics import add_radar_tercile_group


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
    twcrps_alpha: float = 1.0,
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
        alpha=twcrps_alpha,
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

    out["twcrps_sum"] = tw_discrete["twcrps_discrete_sum"]
    out["twcrps_mean_obs"] = tw_discrete["twcrps_discrete_mean_obs"]
    out["twcrps_mean_obs_thr"] = tw_discrete["twcrps_discrete_mean_obs_thr"]

    out["twcrps_threshold_q95"] = u0
    out["twcrps_n_thresholds"] = int(len(tw_thresholds))
    out["twcrps_threshold_min"] = float(np.min(tw_thresholds))
    out["twcrps_threshold_max"] = float(np.max(tw_thresholds))

    out["twcrps_mean"] = out["twcrps_sum"] / len(tw_thresholds) if len(tw_thresholds) > 0 else np.nan

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



def cramer_von_mises_from_pit(pit: Sequence[float]) -> float:
    u = np.sort(np.asarray(pit, dtype=float))
    u = u[np.isfinite(u)]
    n = len(u)
    if n == 0:
        return np.nan
    i = np.arange(1, n + 1)
    return float(1.0 / (12.0 * n) + np.sum((u - (2.0 * i - 1.0) / (2.0 * n)) ** 2))


def compute_pit(pred_df: pd.DataFrame) -> np.ndarray:
    return np.clip(
        egpd_cdf(
            pred_df["Y_obs"].to_numpy(float),
            pred_df["sigma"].to_numpy(float),
            pred_df["kappa"].to_numpy(float),
            pred_df["xi"].to_numpy(float),
        ),
        1e-12,
        1 - 1e-12,
    )


def transformed_exponential_pit(pred_df: pd.DataFrame) -> np.ndarray:
    pit = compute_pit(pred_df)
    z = -np.log(1 - pit)
    return z[np.isfinite(z)]
    

# Scores and skill scores
def twcrps_alpha_outputs(
    pred_df: pd.DataFrame,
    alpha: float = 1.0,
    q_low: float = 0.95,
    q_high: float = 0.995,
    n_thresholds: int = 24,
    thresholds: np.ndarray | None = None,
    normalize_weights: bool = False,
) -> dict:
    y = pred_df["Y_obs"].to_numpy(float)
    y_pos = y[np.isfinite(y) & (y > 0)]

    if len(y_pos) < 20:
        return {
            "twcrps_sum": np.nan,
            "twcrps_mean": np.nan,
            "twcrps_thresholds": np.nan,
            "twcrps_n_obs": len(y),
        }

    if thresholds is None:
        thresholds = make_rain_twcrps_thresholds(
            y,
            q_low=q_low,
            q_high=q_high,
            n_thresholds=n_thresholds,
        )

    thresholds = np.asarray(thresholds, dtype=float)
    thresholds = np.unique(thresholds[np.isfinite(thresholds)])

    if len(thresholds) == 0:
        return {
            "twcrps_sum": np.nan,
            "twcrps_mean": np.nan,
            "twcrps_thresholds": 0,
            "twcrps_n_obs": len(y),
        }

    u0 = float(np.nanquantile(y_pos, q_low))
    if (not np.isfinite(u0)) or u0 <= 0:
        u0 = float(np.nanmin(thresholds[thresholds > 0]))

    weights = twcrps_weight_rain_power(
        thresholds,
        u0=u0,
        alpha=alpha,
        normalize=normalize_weights,
    )

    out = twcrps_discrete_sum_egpd(
        y=y,
        sigma=pred_df["sigma"].to_numpy(float),
        kappa=pred_df["kappa"].to_numpy(float),
        xi=pred_df["xi"].to_numpy(float),
        thresholds=thresholds,
        weights=weights,
    )

    sum_keys = ["twcrps_discrete_sum_obs_thr", "twcrps_sum", "twcrps_sum_obs_thr"]
    mean_keys = ["twcrps_discrete_mean_obs_thr", "twcrps_mean", "twcrps_mean_obs_thr"]

    tw_sum = next((out[k] for k in sum_keys if k in out), None)
    tw_mean = next((out[k] for k in mean_keys if k in out), None)

    if tw_sum is None and tw_mean is not None:
        tw_sum = float(tw_mean) * len(y) * len(thresholds)
    if tw_mean is None and tw_sum is not None:
        tw_mean = float(tw_sum) / (len(y) * len(thresholds))

    return {
        "twcrps_sum": float(tw_sum),
        "twcrps_mean": float(tw_mean),
        "twcrps_thresholds": int(len(thresholds)),
        "twcrps_n_obs": int(len(y)),
    }


def score_one_prediction_table(pred_df: pd.DataFrame, alpha: float = 1.0) -> dict:
    y = pred_df["Y_obs"].to_numpy(float)
    sigma = pred_df["sigma"].to_numpy(float)
    kappa = pred_df["kappa"].to_numpy(float)
    xi = pred_df["xi"].to_numpy(float)

    pit = np.clip(egpd_cdf(y, sigma, kappa, xi), 1e-12, 1 - 1e-12)
    z = -np.log(1 - pit)

    metrics = summarize_distribution_metrics(y, sigma, kappa, xi)
    smad = smad_exponential_margins(y, sigma, kappa, xi, p1=0.95)
    tw = twcrps_alpha_outputs(pred_df, alpha=alpha, normalize_weights=False)

    return {
        "n": len(pred_df),
        "crps_mean": metrics.get("crps_mean", np.nan),
        "twcrps_sum": tw["twcrps_sum"],
        "twcrps_mean": tw["twcrps_mean"],
        "twcrps_n_obs": tw["twcrps_n_obs"],
        "twcrps_thresholds": tw["twcrps_thresholds"],
        "smad": smad["smad"],
        "pit_mean": float(np.mean(pit)),
        "pit_var": float(np.var(pit, ddof=1)),
        "pit_cvm": cramer_von_mises_from_pit(pit),
        "exp_mean": float(np.mean(z)),
        "exp_var": float(np.var(z, ddof=1)),
        "kappa_median": float(np.nanmedian(kappa)),
        "kappa_q95": float(np.nanquantile(kappa, 0.95)),
        "kappa_q99": float(np.nanquantile(kappa, 0.99)),
        "kappa_max": float(np.nanmax(kappa)),
        "prop_kappa_gt_1": float(np.mean(kappa > 1.0)),
        "prop_kappa_gt_2": float(np.mean(kappa > KAPPA_LIMIT)),
        "kappa_excess_mean": float(np.mean(np.maximum(kappa - KAPPA_LIMIT, 0.0))),
    }


def score_loso_by_station(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, station), d in pred_df.groupby(["model", "left_out_station"]):
        rows.append({
            "model": model,
            "left_out_station": station,
            **score_one_prediction_table(d, alpha=1.0),
        })
    return pd.DataFrame(rows)


def summarize_loso_scores(loso_scores: pd.DataFrame) -> pd.DataFrame:
    summary = (
        loso_scores
        .groupby("model")
        .agg(
            n_sites=("left_out_station", "nunique"),
            n_obs=("n", "sum"),
            twcrps_sum_total=("twcrps_sum", "sum"),
            twcrps_sum_mean_site=("twcrps_sum", "mean"),
            twcrps_sum_sd_site=("twcrps_sum", "std"),
            twcrps_mean_mean_site=("twcrps_mean", "mean"),
            crps_mean=("crps_mean", "mean"),
            crps_sd=("crps_mean", "std"),
            smad_mean=("smad", "mean"),
            smad_sd=("smad", "std"),
            pit_cvm_mean=("pit_cvm", "mean"),
            pit_cvm_sd=("pit_cvm", "std"),
            pit_mean_mean=("pit_mean", "mean"),
            pit_var_mean=("pit_var", "mean"),
            kappa_q99_mean=("kappa_q99", "mean"),
            prop_kappa_gt_2_mean=("prop_kappa_gt_2", "mean"),
            kappa_excess_mean=("kappa_excess_mean", "mean"),
        )
        .reset_index()
    )

    for col in [
        "twcrps_sum_total",
        "twcrps_sum_mean_site",
        "twcrps_mean_mean_site",
        "crps_mean",
        "smad_mean",
        "pit_cvm_mean",
        "kappa_q99_mean",
        "prop_kappa_gt_2_mean",
    ]:
        if col in summary.columns:
            best = summary[col].min(skipna=True)
            summary[f"{col}_delta"] = summary[col] - best
            summary[f"{col}_rel_delta_pct"] = 100.0 * (summary[col] / best - 1.0) if best != 0 else np.nan

    return summary.sort_values("twcrps_sum_total").reset_index(drop=True)


def add_skill_scores_vs_reference(
    scores_df: pd.DataFrame,
    group_cols: list[str] | None = None,
    ref_model: str = REFERENCE_MODEL,
    score_cols: list[str] | None = None,
) -> pd.DataFrame:
    if group_cols is None:
        group_cols = []
    if score_cols is None:
        score_cols = ["crps_mean", "twcrps_sum", "twcrps_mean"]

    skill_names = {
        "crps_mean": "crps_skill",
        "twcrps_sum": "twcrps_skill",
        "twcrps_mean": "twcrps_mean_skill",
        "smad": "smad_skill",
    }

    out = scores_df.copy()

    if len(group_cols) == 0:
        ref_rows = out[out["model"] == ref_model]
        if len(ref_rows) != 1:
            raise ValueError(f"Expected exactly one reference row for {ref_model}, got {len(ref_rows)}.")

        ref_row = ref_rows.iloc[0]
        for c in score_cols:
            ref_value = ref_row[c]
            out[f"{c}_ref"] = ref_value
            out[skill_names.get(c, f"{c}_skill")] = np.where(
                ref_value > 0,
                1.0 - out[c] / ref_value,
                np.nan,
            )
        return out

    ref = (
        out[out["model"] == ref_model]
        [group_cols + score_cols]
        .rename(columns={c: f"{c}_ref" for c in score_cols})
    )

    out = out.merge(ref, on=group_cols, how="left")

    for c in score_cols:
        ref_c = f"{c}_ref"
        out[skill_names.get(c, f"{c}_skill")] = np.where(
            out[ref_c] > 0,
            1.0 - out[c] / out[ref_c],
            np.nan,
        )

    return out


def score_by_radar_tercile(pred_df: pd.DataFrame, site: str) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        if model not in pred_df["model"].unique():
            continue
        d = add_radar_tercile_group(
            pred_df,
            site=site,
            model=model,
            station_col="left_out_station",
            use_dt0h_only=True,
            radar_summary="mean",
        )
        for tercile, g in d.groupby("radar_tercile", observed=True):
            if len(g) < 20:
                continue
            rows.append({
                "model": model,
                "site": site,
                "radar_tercile": str(tercile),
                **score_one_prediction_table(g, alpha=1.0),
            })
    return pd.DataFrame(rows)