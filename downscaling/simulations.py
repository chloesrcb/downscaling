from __future__ import annotations

import numpy as np
import pandas as pd

from downscaling.egpd import qegpd


def simulate_positive_chunk(df_chunk: pd.DataFrame, n_sim: int, seed: int) -> np.ndarray:
    """Simulate positive rainfall intensities for one chunk."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=(len(df_chunk), n_sim))

    y = qegpd(
        prob=u,
        sigma=df_chunk["sigma"].to_numpy(float)[:, None],
        kappa=df_chunk["kappa"].to_numpy(float)[:, None],
        xi=df_chunk["xi"].to_numpy(float)[:, None],
    )

    return y.astype(np.float32)


def add_distribution_indicators(
    grid_pred: pd.DataFrame,
    n_sim_mean: int = 200,
    batch_size: int = 100_000,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Add 5-min distribution summaries.

    Quantiles are computed directly from the fitted EGPD quantile function.
    Means are estimated by Monte Carlo in batches to avoid storing a huge
    simulation matrix.
    """
    out = grid_pred.copy()

    sigma = out["sigma"].to_numpy(float)
    kappa = out["kappa"].to_numpy(float)
    xi = out["xi"].to_numpy(float)

    for q in [0.50, 0.90, 0.95, 0.99]:
        col = f"rain_pos_q{int(q * 100):02d}"
        out[col] = qegpd(q, sigma=sigma, kappa=kappa, xi=xi)
        out[f"log_{col}"] = np.log1p(out[col])

    rain_pos_mean = np.empty(len(out), dtype=np.float32)
    log_rain_pos_mean = np.empty(len(out), dtype=np.float32)
    mean_above_row_q95 = np.empty(len(out), dtype=np.float32)
    log_mean_above_row_q95 = np.empty(len(out), dtype=np.float32)

    q95 = out["rain_pos_q95"].to_numpy(float)

    for start in range(0, len(out), batch_size):
        end = min(start + batch_size, len(out))
        chunk = out.iloc[start:end].copy()

        y = simulate_positive_chunk(
            df_chunk=chunk,
            n_sim=n_sim_mean,
            seed=seed + start,
        )

        rain_pos_mean[start:end] = np.mean(y, axis=1)
        log_rain_pos_mean[start:end] = np.mean(np.log1p(y), axis=1)

        q95_chunk = q95[start:end][:, None]
        y_above = np.where(y > q95_chunk, y, np.nan)

        mean_above_row_q95[start:end] = np.nanmean(y_above, axis=1)
        log_mean_above_row_q95[start:end] = np.nanmean(np.log1p(y_above), axis=1)

        print(f"Positive-intensity summaries: {end:,}/{len(out):,}")

    out["rain_pos_mean"] = rain_pos_mean
    out["log_rain_pos_mean"] = log_rain_pos_mean
    out["mean_above_row_q95"] = mean_above_row_q95
    out["log_mean_above_row_q95"] = log_mean_above_row_q95

    out["rain_5min_expected"] = out["p_occ_hat"] * out["rain_pos_mean"]

    return out


def summarize_month_maps(
    grid_pred: pd.DataFrame,
    selection_label: str,
    month_label: str,
    mask=None,
) -> pd.DataFrame:
    """
    Summarise 5-min predictions by grid cell.

    The monthly accumulation is a sum over all selected 5-min time steps.
    """
    if mask is None:
        gp = grid_pred.copy()
    else:
        gp = grid_pred.loc[np.asarray(mask, dtype=bool)].copy()

    summary = (
        gp.groupby(["grid_id", "lon_Y", "lat_Y", "x_l93", "y_l93"])
        .agg(
            prob_occ_5min_mean=("p_occ_hat", "mean"),
            expected_wet_5min_steps=("p_occ_hat", "sum"),
            rain_monthly_sum_expected=("rain_5min_expected", "sum"),
            rain_pos_mean=("rain_pos_mean", "mean"),
            rain_pos_q50=("rain_pos_q50", "mean"),
            rain_pos_q90=("rain_pos_q90", "mean"),
            rain_pos_q95=("rain_pos_q95", "mean"),
            rain_pos_q99=("rain_pos_q99", "mean"),
            mean_above_row_q95=("mean_above_row_q95", "mean"),
            log_rain_pos_mean=("log_rain_pos_mean", "mean"),
            log_rain_pos_q50=("log_rain_pos_q50", "mean"),
            log_rain_pos_q90=("log_rain_pos_q90", "mean"),
            log_rain_pos_q95=("log_rain_pos_q95", "mean"),
            log_rain_pos_q99=("log_rain_pos_q99", "mean"),
            log_mean_above_row_q95=("log_mean_above_row_q95", "mean"),
            sigma_mean=("sigma", "mean"),
            kappa_mean=("kappa", "mean"),
            radar_mean_dt0h=("radar_mean_dt0h", "mean"),
            radar_max_dt0h=("radar_max_dt0h", "mean"),
            radar_central_dt0h=("radar_central_dt0h", "mean"),
        )
        .reset_index()
    )

    summary["month"] = month_label
    summary["selection"] = selection_label

    print(f"\nSummary: {selection_label}")
    print(summary.describe())

    return summary


def longest_run_bool(x: np.ndarray) -> int:
    """Longest run of True values."""
    max_run = 0
    cur = 0

    for v in np.asarray(x, dtype=bool):
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0

    return max_run


def mean_run_bool(x: np.ndarray) -> float:
    """Mean run length of True values."""
    runs = []
    cur = 0

    for v in np.asarray(x, dtype=bool):
        if v:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0

    if cur > 0:
        runs.append(cur)

    return 0.0 if len(runs) == 0 else float(np.mean(runs))


def summarize_dry_periods(
    grid_pred: pd.DataFrame,
    n_sim_dry: int = 200,
    seed: int = 123,
    month_label: str | None = None,
) -> pd.DataFrame:
    """
    Simulate occurrence only to summarise dry periods.

    A dry day is defined as a day with no simulated 5-min rainfall occurrence.
    """
    rng = np.random.default_rng(seed)

    meta = grid_pred[["grid_id", "time", "lon_Y", "lat_Y", "x_l93", "y_l93", "p_occ_hat"]].copy()
    meta["date"] = pd.to_datetime(meta["time"], utc=True).dt.floor("D")

    rows = []

    for grid_id, idx in meta.groupby("grid_id").indices.items():
        g = meta.iloc[idx].copy()
        p = g["p_occ_hat"].to_numpy(float)

        dry_day_fractions = []
        n_dry_days = []
        mean_dry_runs = []
        max_dry_runs = []
        wet_5min_fractions = []

        for _ in range(n_sim_dry):
            wet = rng.uniform(size=len(p)) < p
            dry_step = ~wet

            g_s = g[["date"]].copy()
            g_s["dry_step"] = dry_step
            g_s["wet_step"] = wet

            daily = (
                g_s.groupby("date")
                .agg(
                    dry_day=("dry_step", "all"),
                    wet_day=("wet_step", "any"),
                )
                .reset_index()
                .sort_values("date")
            )

            dry_seq = daily["dry_day"].to_numpy(bool)

            dry_day_fractions.append(np.mean(dry_seq))
            n_dry_days.append(np.sum(dry_seq))
            mean_dry_runs.append(mean_run_bool(dry_seq))
            max_dry_runs.append(longest_run_bool(dry_seq))
            wet_5min_fractions.append(np.mean(wet))

        rows.append({
            "grid_id": grid_id,
            "lon_Y": g["lon_Y"].iloc[0],
            "lat_Y": g["lat_Y"].iloc[0],
            "x_l93": g["x_l93"].iloc[0],
            "y_l93": g["y_l93"].iloc[0],
            "wet_5min_fraction": float(np.mean(wet_5min_fractions)),
            "dry_day_fraction": float(np.mean(dry_day_fractions)),
            "n_dry_days_mean": float(np.mean(n_dry_days)),
            "n_dry_days_q05": float(np.quantile(n_dry_days, 0.05)),
            "n_dry_days_q95": float(np.quantile(n_dry_days, 0.95)),
            "mean_consecutive_dry_days": float(np.mean(mean_dry_runs)),
            "max_consecutive_dry_days_mean": float(np.mean(max_dry_runs)),
            "max_consecutive_dry_days_q95": float(np.quantile(max_dry_runs, 0.95)),
        })

    summary_dry = pd.DataFrame(rows)
    if month_label is not None:
        summary_dry["month"] = month_label
    summary_dry["selection"] = "dry_periods_occurrence_simulation"

    print("\nDry-period summary:")
    print(summary_dry.describe())

    return summary_dry
