from __future__ import annotations

import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

PROBS_QQ = np.concatenate([
    np.linspace(0.01, 0.90, 90),
    np.linspace(0.91, 0.99, 18),
    np.array([0.995, 0.999]),
])

import matplotlib.pyplot as plt
from downscaling.settings import MODEL_ORDER, MODEL_ORDER_NO_REF, REFERENCE_MODEL

from downscaling.egpd import qegpd



PLOT_DPI = 180
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 11
RESPONSE_LABEL = r"Local rainfall $X_{\mathbf{s},t}$"
POSITIVE_RESPONSE_LABEL = r"Positive local rainfall $X_{\mathbf{s},t}$"
LOG_RESPONSE_LABEL = r"$\log(1 + X_{\mathbf{s},t})$"


PALETTE = {
    "Stationary EGPD": "#8B0000",  # darkred
    "Stationary": "#8B0000",
    "GLM": "#F28E2B",              # orange
    "GAM": "#D45087",              # rose
    "NN": "#008080",               # teal
    "Observed": "#8B0000",
    "Simulated": "#008080",
    "CRPS": "#D45087",
    "twCRPS": "#008080",
    "CRPSS": "#D45087",
    "twCRPSS": "#008080",
    "darkred": "#8B0000",
}

SCORE_LABELS = {
    "twcrps_sum": "twCRPS",
    "twcrps_mean": "Mean twCRPS",
    "crps_mean": "Mean CRPS",
    "crps_sum": "CRPS",
    "smad": "sMAD",
    "pit_cvm": "PIT CvM",
    "kappa_q99": r"99th percentile of $\kappa$",
    "prop_kappa_gt_2": r"Proportion of $\kappa > 2$",
    "crps_skill": "SS CRPS",
    "twcrps_skill": "SS twCRPS",
}

import matplotlib.colors as mcolors
RAIN_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "rain_blue",
    ["#F7FBFF", "#DEEBF7", "#9ECAE1", "#4292C6", "#08519C"],
    N=256,
)

DRY_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "dry_orange",
    ["#FFF7EC", "#FDD49E", "#FDBB84", "#EF6548", "#990000"],
    N=256,
)

PARAM_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "param_purple",
    ["#F7FCFD", "#E0ECF4", "#BFD3E6", "#9EBCDA", "#8C96C6", "#88419D"],
    N=256,
)

def savefig(fig, name: str, out_dir: Path) -> None:
    save_png(fig, out_dir / name)

def configure_plot_style() -> None:
    """Apply a clean style for saved figures."""
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.labelsize": AXIS_LABEL_FONTSIZE,
            "xtick.labelsize": TICK_LABEL_FONTSIZE,
            "ytick.labelsize": TICK_LABEL_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "figure.titlesize": AXIS_LABEL_FONTSIZE,
            "savefig.dpi": PLOT_DPI,
            "savefig.transparent": True,
            "savefig.bbox": "tight",
        }
    )


def clean_figure(fig) -> None:
    """Remove frames and make the figure background transparent."""
    fig.patch.set_alpha(0.0)
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")

    for ax in fig.axes:
        ax.set_facecolor("none")
        ax.set_frame_on(False)
        ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
        ax.xaxis.label.set_size(AXIS_LABEL_FONTSIZE)
        ax.yaxis.label.set_size(AXIS_LABEL_FONTSIZE)
        ax.title.set_size(AXIS_LABEL_FONTSIZE)
        for spine in ax.spines.values():
            spine.set_visible(False)


def save_png(fig, path: str | os.PathLike, dpi: int = PLOT_DPI) -> None:
    """Save a transparent, compressed PNG."""
    clean_figure(fig)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        transparent=True,
        facecolor="none",
        edgecolor="none",
        pil_kwargs={"compress_level": 9},
    )


def pretty_predictor_name(name: str) -> str:
    """Convert technical predictor names into notation consistent with the text."""
    m = re.match(r"^X_p(\d{2})_dt(-1h|0h|\+1h)$", name)
    if m is not None:
        pixel = int(m.group(1))
        lag = m.group(2)
        lag_index = {"-1h": "-1", "0h": "0", "+1h": "1"}.get(lag, lag)
        return rf"COMEPHORE predictor $C_{{{pixel},{lag_index}}}$"

    replacements = {
        "radar_max": r"$\max(\mathbf{C}^{\mathrm{cube}}_{\mathbf{s},t})$",
        "radar_mean": r"$\mathrm{mean}(\mathbf{C}^{\mathrm{cube}}_{\mathbf{s},t})$",
        "radar_sum": r"$\sum \mathbf{C}^{\mathrm{cube}}_{\mathbf{s},t}$",
        "radar_max_dt0h": r"$\max C_{j,0}(\mathbf{s},t)$",
        "radar_mean_dt0h": r"$\mathrm{mean} C_{j,0}(\mathbf{s},t)$",
        "radar_sum_dt0h": r"$\sum C_{j,0}(\mathbf{s},t)$",
        # "radar_central_dt0h": r"$C_{\mathrm{central},0}(\mathbf{s},t)$",
        "tod_sin": "Time of day, sine",
        "tod_cos": "Time of day, cosine",
        "doy_sin": "Day of year, sine",
        "doy_cos": "Day of year, cosine",
        "month_sin": "Month, sine",
        "month_cos": "Month, cosine",
        "lat_Y": r"Gauge latitude",
        "lon_Y": r"Gauge longitude",
        "lat_X": r"COMEPHORE pixel latitude",
        "lon_X": r"COMEPHORE pixel longitude",
        "Y_obs": RESPONSE_LABEL,
        "q50_pred": r"Predicted median of $X_{\mathbf{s},t}$",
        "q75_pred": r"Predicted 75% quantile of $X_{\mathbf{s},t}$",
        "q90_pred": r"Predicted 90% quantile of $X_{\mathbf{s},t}$",
        "q95_pred": r"Predicted 95% quantile of $X_{\mathbf{s},t}$",
        "q99_pred": r"Predicted 99% quantile of $X_{\mathbf{s},t}$",
        "mean_pred": r"Predicted mean of $X_{\mathbf{s},t}$",
    }

    return replacements.get(name, name)



def compact_model_name(name: str) -> str:
    s = str(name)
    if "Stationary" in s or "EGPD" in s:
        return "Stationary"
    if "GLM" in s:
        return "GLM"
    if "GAM" in s:
        return "GAM"
    if "NN" in s:
        return "NN"
    return s


def model_color(model: str) -> str:
    return PALETTE.get(model, PALETTE.get(compact_model_name(model), "#333333"))



# Plotting score summaries
def plot_loso_score_boxplot(score_df: pd.DataFrame, score_col: str, out_dir: Path) -> None:
    plot_df = score_df.copy()
    plot_df["model_short"] = plot_df["model"].map(compact_model_name)
    models = [compact_model_name(m) for m in MODEL_ORDER if compact_model_name(m) in plot_df["model_short"].unique()]

    data = [plot_df.loc[plot_df["model_short"] == m, score_col].dropna().to_numpy() for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, tick_labels=models, showmeans=True, patch_artist=True)

    for patch, model in zip(bp["boxes"], models):
        patch.set_facecolor(model_color(model))
        patch.set_alpha(0.55)
        patch.set_edgecolor("#333333")

    for med in bp["medians"]:
        med.set_color("#8B0000")
        med.set_linewidth(1.6)

    ax.set_ylabel(SCORE_LABELS.get(score_col, score_col))
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    savefig(fig, f"loso_boxplot_{score_col}.png", out_dir=out_dir)
    plt.show()


def plot_skill_score_boxplots(
    skill_df: pd.DataFrame,
    skill_cols: list[str] | None = None,
    filename: str = "loso_skill_score_boxplots.png",
    out_dir: Path = Path("."),
) -> None:
    if skill_cols is None:
        skill_cols = ["crps_skill", "twcrps_skill"]

    d = skill_df[skill_df["model"] != REFERENCE_MODEL].copy()
    d["model_short"] = d["model"].map(compact_model_name)
    models = [compact_model_name(m) for m in MODEL_ORDER_NO_REF if compact_model_name(m) in d["model_short"].unique()]

    fig, axes = plt.subplots(1, len(skill_cols), figsize=(5.5 * len(skill_cols), 4.6), sharey=True)
    if len(skill_cols) == 1:
        axes = [axes]

    rng = np.random.default_rng(123)

    for ax, skill_col in zip(axes, skill_cols):
        data = [d.loc[d["model_short"] == model, skill_col].dropna().to_numpy() for model in models]
        bp = ax.boxplot(data, tick_labels=models, showmeans=True, patch_artist=True)

        for patch, model in zip(bp["boxes"], models):
            patch.set_facecolor(model_color(model))
            patch.set_alpha(0.55)
            patch.set_edgecolor("#333333")

        # for j, vals in enumerate(data, start=1):
        #     x_jitter = rng.normal(loc=j, scale=0.035, size=len(vals))
        #     ax.scatter(x_jitter, vals, s=20, alpha=0.65, color="#333333", linewidth=0)

        for med in bp["medians"]:
            med.set_color("#8B0000")
            med.set_linewidth(1.6)

        ax.axhline(0.0, linestyle="--", linewidth=1.2, color="#8B0000")
        # ax.set_title(SCORE_LABELS.get(skill_col, skill_col))
        ax.set_ylabel("Skill score vs Stationary EGPD")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    savefig(fig, filename, out_dir=out_dir)
    plt.show()


def plot_skill_score_heatmap(
    skill_df: pd.DataFrame,
    skill_col: str = "twcrps_skill",
    filename: str = "loso_twcrps_skill_heatmap_by_site.png",
    out_dir: Path = Path("."),
) -> None:
    d = skill_df[skill_df["model"] != REFERENCE_MODEL].copy()

    heat = d.pivot(index="left_out_station", columns="model", values=skill_col)
    heat = heat[[m for m in MODEL_ORDER_NO_REF if m in heat.columns]]

    values = heat.to_numpy(float)
    vmax = np.nanmax(np.abs(values))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rose_white_teal",
        ["#8B0000", "#F4A6C1", "#FFFFFF", "#7BCDC8", "#008080"],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(7, max(5, 0.35 * len(heat))))
    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_xticklabels([compact_model_name(c) for c in heat.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_yticklabels(heat.index)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(SCORE_LABELS.get(skill_col, skill_col))

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="#222222")

    plt.tight_layout()
    savefig(fig, filename, out_dir=out_dir)
    plt.show()


def plot_global_score_and_skill_summary(
    global_scores_skill: pd.DataFrame,
    filename: str = "loso_global_score_and_skill_summary.png",
    out_dir: Path = Path(".")
) -> None:
    d = global_scores_skill.copy()
    d["model_short"] = d["model"].map(compact_model_name)
    d = d[d["model"] != REFERENCE_MODEL].copy()

    d["CRPS / Stationary"] = d["crps_mean"] / d["crps_mean_ref"]
    d["twCRPS / Stationary"] = d["twcrps_sum"] / d["twcrps_sum_ref"]
    d["CRPSS"] = d["crps_skill"]
    d["twCRPSS"] = d["twcrps_skill"]

    x_models = list(d["model_short"])
    x = np.arange(len(x_models))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    axes[0].bar(
        x - width / 2,
        d.set_index("model_short").loc[x_models, "CRPS / Stationary"],
        width,
        label="CRPS",
        color=PALETTE["CRPS"],
        alpha=0.85,
    )
    axes[0].bar(
        x + width / 2,
        d.set_index("model_short").loc[x_models, "twCRPS / Stationary"],
        width,
        label="twCRPS",
        color=PALETTE["twCRPS"],
        alpha=0.85,
    )
    axes[0].axhline(1.0, linestyle="--", linewidth=1.2, color="#8B0000")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_models, rotation=30, ha="right")
    axes[0].set_ylabel("Score ratio to Stationary EGPD")
    axes[0].set_title("Relative scores")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(
        x - width / 2,
        d.set_index("model_short").loc[x_models, "CRPSS"],
        width,
        label="CRPSS",
        color=PALETTE["CRPSS"],
        alpha=0.85,
    )
    axes[1].bar(
        x + width / 2,
        d.set_index("model_short").loc[x_models, "twCRPSS"],
        width,
        label="twCRPSS",
        color=PALETTE["twCRPSS"],
        alpha=0.85,
    )
    axes[1].axhline(0.0, linestyle="--", linewidth=1.2, color="#8B0000")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_models, rotation=30, ha="right")
    axes[1].set_ylabel("Skill score vs Stationary EGPD")
    axes[1].set_title("Skill scores")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    savefig(fig, filename, out_dir=out_dir)
    plt.show()


# Plotting PIT, QQ, simulation diagnostics
def plot_pit_histogram(pred_df: pd.DataFrame, model: str, 
                       out_dir: Path = Path(".")) -> None:
    from downscaling.scores import compute_pit

    d = pred_df[pred_df["model"] == model].copy()
    pit = compute_pit(d)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(pit, bins=20, density=True, alpha=0.70, color=model_color(model))
    ax.axhline(1.0, linestyle="--", linewidth=1.2, color="#8B0000")
    ax.set_xlabel("PIT")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig(fig, f"loso_pit_histogram_{compact_model_name(model)}.png", out_dir=out_dir)
    plt.show()


def plot_exponential_qq_all_models(pred_df: pd.DataFrame, 
                                   out_dir: Path = Path(".")) -> None:
    from downscaling.scores import transformed_exponential_pit

    fig, ax = plt.subplots(figsize=(7, 7))

    for model in [m for m in MODEL_ORDER if m in pred_df["model"].unique()]:
        d = pred_df[pred_df["model"] == model].copy()
        z = transformed_exponential_pit(d)
        q_obs = np.quantile(z, PROBS_QQ)
        q_exp = -np.log(1 - PROBS_QQ)

        ax.plot(
            q_obs,
            q_exp,
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=compact_model_name(model),
            color=model_color(model),
            alpha=0.9,
        )

    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.2, color="#8B0000")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Transformed PIT quantiles")
    ax.set_ylabel("Theoretical exponential quantiles")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    savefig(fig, "loso_exponential_qq_all_models.png", out_dir=out_dir)
    plt.show()


def plot_exponential_qq_all_models_zoom(
    pred_df: pd.DataFrame,
    pmin_zoom: float = 0.90,
    out_dir: Path = Path(".")
) -> None:
    from downscaling.scores import transformed_exponential_pit

    probs_zoom = PROBS_QQ[PROBS_QQ >= pmin_zoom]

    fig, ax = plt.subplots(figsize=(7, 7))

    for model in [m for m in MODEL_ORDER if m in pred_df["model"].unique()]:
        d = pred_df[pred_df["model"] == model].copy()
        z = transformed_exponential_pit(d)

        q_obs = np.quantile(z, probs_zoom)
        q_exp = -np.log(1 - probs_zoom)

        ax.plot(
            q_obs,
            q_exp,
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=compact_model_name(model),
            color=model_color(model),
            alpha=0.9,
        )

    lim_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
    lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1])

    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        linestyle="--",
        linewidth=1.2,
        color="#8B0000",
    )

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    ax.set_xlabel(f"Transformed PIT quantiles, p ≥ {pmin_zoom}")
    ax.set_ylabel("Theoretical exponential quantiles")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    savefig(fig, f"loso_exponential_qq_all_models_zoom_p{int(100*pmin_zoom)}.png", out_dir=out_dir)
    plt.show()


def plot_exponential_qq_all_models_by_site(
    pred_df: pd.DataFrame,
    site: str,
    pmin_zoom: float = 0.90,
    out_dir: Path = Path(".")
) -> None:
    from downscaling.scores import transformed_exponential_pit

    probs_zoom = PROBS_QQ[PROBS_QQ >= pmin_zoom]

    fig, ax = plt.subplots(figsize=(7, 7))

    for model in [m for m in MODEL_ORDER if m in pred_df["model"].unique()]:
        d = pred_df[
            (pred_df["model"] == model)
            & (pred_df["left_out_station"] == site)
        ].copy()

        if len(d) < 20:
            continue

        z = transformed_exponential_pit(d)

        q_obs = np.quantile(z, probs_zoom)
        q_exp = -np.log(1 - probs_zoom)

        ax.plot(
            q_obs,
            q_exp,
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=compact_model_name(model),
            color=model_color(model),
            alpha=0.9,
        )

    lim_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
    lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1])

    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        linestyle="--",
        linewidth=1.2,
        color="#8B0000",
    )

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    ax.set_xlabel(f"Transformed PIT quantiles, p ≥ {pmin_zoom}")
    ax.set_ylabel("Theoretical exponential quantiles")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    savefig(
        fig,
        f"loso_exponential_qq_all_models_site_{site}_zoom_p{int(100*pmin_zoom)}.png",
        out_dir=out_dir
    )
    plt.show()





def plot_exponential_qq_all_models_by_radar_tercile(
    pred_df: pd.DataFrame,
    site: str,
    use_dt0h_only: bool = True,
    radar_summary: str = "mean",
) -> None:
    from downscaling.diagnostics import add_radar_tercile_group
    from downscaling.scores import transformed_exponential_pit

    tercile_order = ["Low radar", "Medium radar", "High radar"]

    for tercile in tercile_order:
        fig, ax = plt.subplots(figsize=(7, 7))

        for model in [m for m in MODEL_ORDER if m in pred_df["model"].unique()]:
            d = add_radar_tercile_group(
                pred_df,
                site=site,
                model=model,
                station_col="left_out_station",
                use_dt0h_only=use_dt0h_only,
                radar_summary=radar_summary,
            )
            d = d[d["radar_tercile"] == tercile].copy()
            if len(d) < 20:
                continue

            z = transformed_exponential_pit(d)
            q_obs = np.quantile(z, PROBS_QQ)
            q_exp = -np.log(1 - PROBS_QQ)

            ax.plot(
                q_obs,
                q_exp,
                marker="o",
                markersize=3,
                linewidth=1.2,
                label=compact_model_name(model),
                color=model_color(model),
                alpha=0.9,
            )

        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.2, color="#8B0000")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Transformed PIT quantiles")
        ax.set_ylabel("Theoretical exponential quantiles")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        tercile_name = str(tercile).replace(" ", "_").replace("-", "_").lower()
        savefig(fig, f"loso_exponential_qq_all_models_site_{site}_{tercile_name}.png", out_dir=out_dir)
        plt.show()


def plot_observed_vs_simulated_density(
    pred_df: pd.DataFrame,
    model: str,
    n_sim_per_obs: int = 50,
    seed: int = 123,
    out_dir: Path = Path(".")
) -> None:
    rng = np.random.default_rng(seed)
    d = pred_df[pred_df["model"] == model].copy()

    u = rng.uniform(size=(len(d), n_sim_per_obs))
    y_sim = qegpd(
        prob=u,
        sigma=d["sigma"].to_numpy(float)[:, None],
        kappa=d["kappa"].to_numpy(float)[:, None],
        xi=d["xi"].to_numpy(float)[:, None],
    ).ravel()
    y_obs = d["Y_obs"].to_numpy(float)

    upper = np.nanquantile(np.r_[y_obs, y_sim], 0.995)
    bins = np.linspace(0, upper, 70)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(y_obs, bins=bins, density=True, alpha=0.45, label="Observed", color=PALETTE["Observed"])
    ax.hist(y_sim, bins=bins, density=True, alpha=0.45, label="Simulated", color=PALETTE["Simulated"])
    ax.set_xlabel(POSITIVE_RESPONSE_LABEL)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    savefig(fig, f"loso_observed_vs_simulated_{compact_model_name(model)}.png", out_dir=out_dir)
    plt.show()


def plot_survival_observed_vs_simulated(
    pred_df: pd.DataFrame,
    model: str,
    n_sim_per_obs: int = 50,
    seed: int = 123,
    out_dir: Path = Path(".")
) -> None:
    rng = np.random.default_rng(seed)
    d = pred_df[pred_df["model"] == model].copy()

    y_obs = d["Y_obs"].to_numpy(float)
    u = rng.uniform(size=(len(d), n_sim_per_obs))
    y_sim = qegpd(
        prob=u,
        sigma=d["sigma"].to_numpy(float)[:, None],
        kappa=d["kappa"].to_numpy(float)[:, None],
        xi=d["xi"].to_numpy(float)[:, None],
    ).ravel()

    y_grid = np.quantile(y_obs[y_obs > 0], np.linspace(0.50, 0.995, 80))
    surv_obs = np.array([(y_obs > yy).mean() for yy in y_grid])
    surv_sim = np.array([(y_sim > yy).mean() for yy in y_grid])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(y_grid, surv_obs, marker="o", markersize=3, label="Observed", color=PALETTE["Observed"])
    ax.plot(y_grid, surv_sim, marker="o", markersize=3, label="Simulated", color=PALETTE["Simulated"])
    ax.set_yscale("log")
    ax.set_xlabel(POSITIVE_RESPONSE_LABEL)
    ax.set_ylabel("Survival probability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    savefig(fig, f"loso_survival_{compact_model_name(model)}.png", out_dir=out_dir)
    plt.show()


def plot_survival_observed_vs_all_models(
    pred_df: pd.DataFrame,
    models: list[str] | None = None,
    n_sim_per_obs: int = 50,
    seed: int = 123,
    out_dir: Path = Path(".")
) -> None:
    rng = np.random.default_rng(seed)

    if models is None:
        models = list(pred_df["model"].unique())

    # Observations: same for all models, take first model subset
    d0 = pred_df[pred_df["model"] == models[0]].copy()
    y_obs = d0["Y_obs"].to_numpy(float)

    y_grid = np.quantile(
        y_obs[y_obs > 0],
        np.linspace(0.50, 0.995, 80)
    )

    surv_obs = np.array([(y_obs > yy).mean() for yy in y_grid])

    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(
        y_grid,
        surv_obs,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Observed",
        color="black",
    )

    for model in models:
        d = pred_df[pred_df["model"] == model].copy()

        u = rng.uniform(size=(len(d), n_sim_per_obs))

        y_sim = qegpd(
            prob=u,
            sigma=d["sigma"].to_numpy(float)[:, None],
            kappa=d["kappa"].to_numpy(float)[:, None],
            xi=d["xi"].to_numpy(float)[:, None],
        ).ravel()

        surv_sim = np.array([(y_sim > yy).mean() for yy in y_grid])
        line_styles = {
            "NN": "-",
            "GAM": "--",
            "GLM": "-.",
            "Stationary": ":",
        }
        ax.plot(
            y_grid,
            surv_sim,
            linewidth=2,
            linestyle=line_styles.get(compact_model_name(model), "-"),
            color=model_color(model),
            label=compact_model_name(model),
        )

    ax.set_yscale("log")
    ax.set_xlabel(POSITIVE_RESPONSE_LABEL)
    ax.set_ylabel("Survival probability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    savefig(fig, "loso_survival_all_models.png", out_dir=out_dir)
    plt.show()

def plot_parameter_boxplots_combined(
    pred_df: pd.DataFrame,
    models: list[str] = ["GLM", "GAM", "NN"],
    params: tuple[str, str] = ("sigma", "kappa"),
    show_outliers: bool = False,
    out_dir: Path = Path(".")
) -> None:

    data_by_param = {p: [] for p in params}
    for p in params:
        for model in models:
            d_model = pred_df[pred_df["model"] == model]
            data_by_param[p].append(d_model[p].dropna().values)

    n_models = len(models)
    positions = []
    current_pos = 1

    for i in range(len(params)):
        positions.extend([current_pos + j for j in range(n_models)])
        current_pos += n_models + 1 

    all_data = data_by_param[params[0]] + data_by_param[params[1]]

    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(
        all_data,
        positions=positions,
        patch_artist=True,
        showfliers=show_outliers,
        widths=0.6,
    )

    for idx, box in enumerate(bp["boxes"]):
        model_idx = idx % n_models
        current_model = models[model_idx]

        box.set_facecolor(model_color(current_model))
        box.set_alpha(0.65)
        box.set_edgecolor("black")

    # Style des médianes
    for median in bp["medians"]:
        median.set_color("darkred")
        median.set_linewidth(2)

    group_centers = [
        1 + (n_models - 1) / 2, 
        (n_models + 1) + 1 + (n_models - 1) / 2,
    ]
    labels = [r"$\widehat{\sigma}_{s,t}$", r"$\widehat{\kappa}_{s,t}$"]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(labels, fontsize=12)

    ax.set_ylabel("Estimated parameter value", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=model_color(m),
            alpha=0.65,
            edgecolor="black",
            label=compact_model_name(m),
        )
        for m in models
    ]
    ax.legend(handles=legend_elements, title="Models", loc="upper right")

    plt.tight_layout()

    models_str = "_".join([compact_model_name(m) for m in models]).lower()
    savefig(fig, f"loso_parameter_boxplots_combined_{models_str}.png", out_dir=out_dir)
    plt.show()


def pivot_grid(summary: pd.DataFrame, value_col: str) -> pd.DataFrame:
    dat = summary[["x_l93", "y_l93", value_col]].copy()
    dat = dat.replace([np.inf, -np.inf], np.nan)
    dat = dat.dropna(subset=["x_l93", "y_l93", value_col])

    return (
        dat.pivot_table(
            index="y_l93",
            columns="x_l93",
            values=value_col,
            aggfunc="mean",
            dropna=False,
        )
        .sort_index()
        .sort_index(axis=1)
    )


def cell_edges_from_centres(v: np.ndarray, grid_res_m: float = None) -> np.ndarray:
    v = np.sort(np.unique(np.asarray(v, dtype=float)))

    if len(v) == 1:
        return np.array([v[0] - grid_res_m / 2, v[0] + grid_res_m / 2])

    mids = (v[:-1] + v[1:]) / 2

    return np.r_[
        v[0] - (v[1] - v[0]) / 2,
        mids,
        v[-1] + (v[-1] - v[-2]) / 2,
    ]


def get_scale(values: np.ndarray, scale: dict | None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    if scale is None:
        scale = {"mode": "quantile", "qmin": 0.02, "qmax": 0.98}

    mode = scale.get("mode", "quantile")

    if mode == "fixed":
        vmin = scale.get("vmin", np.nanmin(values))
        vmax = scale.get("vmax", np.nanmax(values))

    elif mode == "minmax":
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)

    elif mode == "positive_quantile":
        vmin = scale.get("vmin", 0.0)
        vmax = np.nanquantile(values, scale.get("qmax", 0.98))

    elif mode == "quantile":
        vmin = np.nanquantile(values, scale.get("qmin", 0.02))
        vmax = np.nanquantile(values, scale.get("qmax", 0.98))

    else:
        raise ValueError(f"Unknown scale mode: {mode}")

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = np.nanmin(values), np.nanmax(values)

    if vmin == vmax:
        eps = max(abs(vmin) * 0.05, 1e-6)
        vmin -= eps
        vmax += eps

    return vmin, vmax

from matplotlib.ticker import FormatStrFormatter

def plot_heatmap(
    summary: pd.DataFrame,
    value_col: str,
    label: str,
    filename: str,
    cmap=RAIN_CMAP,
    scale: dict | None = None,
    add_gauges: bool = True,
    out_dir: Path = None,
    gauges_plot: pd.DataFrame = None,
    grid_res_m: float = None,
    display_factor: float = 1.0,
    cbar_format: str | None = None,
) -> None:
    pivot = pivot_grid(summary, value_col)

    if pivot.empty:
        print(f"[skip] {value_col}: empty map")
        return

    raw_values = pivot.values.astype(float) * display_factor
    vmin, vmax = get_scale(raw_values, scale)

    print(f"\nPlot {value_col}")
    print((summary[value_col] * display_factor).describe())
    print("scale:", vmin, vmax)

    x_edges = cell_edges_from_centres(pivot.columns.to_numpy(float), grid_res_m=grid_res_m)
    y_edges = cell_edges_from_centres(pivot.index.to_numpy(float), grid_res_m=grid_res_m)

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(raw_values),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)

    if cbar_format is not None:
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(cbar_format))

    if add_gauges:
        ax.scatter(
            gauges_plot["x_l93"],
            gauges_plot["y_l93"],
            s=35,
            c=PALETTE["darkred"],
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    savefig(fig, filename, out_dir=out_dir)
    plt.show()

def plot_summary_set(summary: pd.DataFrame, prefix: str, map_specs: list[dict], out_dir: Path, 
                     gauges_plot: pd.DataFrame = None, grid_res_m: float = None) -> None:
    for spec in map_specs:
        col = spec["col"]

        if col not in summary.columns:
            print(f"[skip] {col}: missing column")
            continue

        plot_heatmap(
            summary=summary,
            value_col=col,
            label=spec["label"],
            filename=f"map_{prefix}_{spec['name']}.png",
            cmap=spec.get("cmap", RAIN_CMAP),
            scale=spec.get("scale"),
            add_gauges=spec.get("add_gauges", True),
            out_dir=out_dir,
            gauges_plot=gauges_plot,
            grid_res_m=grid_res_m,
            display_factor=spec.get("display_factor", 1.0),
            cbar_format=spec.get("cbar_format"),
        )