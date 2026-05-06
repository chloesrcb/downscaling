
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from downscaling.egpd import (
    egpd_left_censored_nll,
    egpd_left_censored_nll_sum,
)
from downscaling.nn import Config, parse_widths, predict_params_on_df_variant
from downscaling.regression import (
    fit_egpd_regression_model,
    predict_egpd_regression_model,
)
from downscaling.scores import (
    smad_exponential_margins,
    summarize_distribution_metrics,
)
from downscaling.stationary import fit_egpd_stationary_direct

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
