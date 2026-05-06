import os
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from downscaling.egpd import egpd_cdf, egpd_mean, qegpd
from downscaling.paths import DIAG_DIR
from downscaling.splits import make_train_valid_test_split

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
        out[f"q{q_int}_pred"] = qegpd(
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
