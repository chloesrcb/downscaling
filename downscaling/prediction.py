
from typing import Optional

import numpy as np
import pandas as pd
import torch

from downscaling.config import (
    CENSOR_THRESHOLD,
    KAPPA_INIT,
    SIGMA_INIT,
    XI_INIT,
)
from downscaling.egpd import (
    egpd_left_censored_nll,
    egpd_left_censored_nll_sum,
)
from downscaling.models import EGPDNNOnlyInputs
from downscaling.nn import Config, parse_widths, predict_params_on_df_variant
from downscaling.features import standardize_train_only
from downscaling.occurrence import train_logit_model, predict_occurrence_probability
from downscaling.regression import (
    fit_egpd_regression_model,
    predict_egpd_regression_model,
)
from downscaling.scores import make_prediction_df, make_test_metric_row
from downscaling.splits import make_single_split_from_train
from downscaling.stationary import fit_egpd_stationary_direct
from downscaling.nn import run_one_nn_variant

@torch.no_grad()
def predict_params(model, X_s=None, X_k=None, offset=None, device=None):
    dev = torch.device(device) if device is not None else next(model.parameters()).device

    def to_tensor_or_none(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.to(device=dev, dtype=torch.float32)
        return torch.as_tensor(x, dtype=torch.float32, device=dev)

    Xs = to_tensor_or_none(X_s)
    Xk = to_tensor_or_none(X_k)
    Off = to_tensor_or_none(offset)

    model.eval()
    pred = model(EGPDNNOnlyInputs(X_s=Xs, X_k=Xk, offset=Off))
    pred = pred.detach().cpu().numpy()

    return {
        "pred_sigma": pred[..., 0],
        "pred_kappa": pred[..., 1],
        "pred_xi": pred[..., 2],
    }




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
        fix_xi=False,
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
        n_train_valid=len(df_train_valid),
        n_test=len(df_test),
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
        n_train_valid=len(df_train_valid),
        n_test=len(df_test),
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
    compute_metrics=True,
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
        block="15D",
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

    if compute_metrics:
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
            n_train_valid=len(df_train_valid),
            n_test=len(df_test),
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
    else:
        row = None
    
    pred = make_prediction_df(
        df_test=df_test,
        model_label=f"NN {x_set_name}",
        sigma_t=sigma_t,
        kappa_t=kappa_t,
        xi_t=xi_t,
    )

    return row, pred, fit


def predict_intensity_on_grid(
    df_model: pd.DataFrame,
    grid_pred: pd.DataFrame,
    x_sets: dict,
    best_params: dict,
    seed: int = 123,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Predict EGPD parameters on the fine grid."""
    _, pred_grid_intensity, _ = fit_predict_nn_test(
        df_train_valid=df_model,
        df_test=grid_pred,
        x_sets=x_sets,
        best_params=best_params,
        seed=seed,
        device=device,
        compute_metrics=False,
    )

    out = grid_pred.copy()
    out["sigma"] = pred_grid_intensity["sigma"].to_numpy(float)
    out["kappa"] = pred_grid_intensity["kappa"].to_numpy(float)
    out["xi"] = pred_grid_intensity["xi"].to_numpy(float)

    return out


def predict_occurrence_on_grid_batched(
    df_occ_train: pd.DataFrame,
    grid_pred: pd.DataFrame,
    x_cols: list[str],
    batch_size_pred: int = 100_000,
    seed: int = 123,
    device: Optional[str] = None,
) -> np.ndarray:
    """Fit the occurrence model and predict 5-min occurrence probabilities by batches."""
    df_train = df_occ_train.copy().reset_index(drop=True)

    train_idx = np.arange(len(df_train))
    df_train_std, means, stds = standardize_train_only(df_train, train_idx, x_cols)

    X_train = df_train_std[x_cols].to_numpy(dtype=np.float32)
    y_train = (df_train_std["Y_obs"].to_numpy(dtype=np.float32) > 0).astype(np.float32)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(df_train_std))
    rng.shuffle(idx)

    n_valid = max(1000, int(0.1 * len(idx)))
    valid_idx = idx[:n_valid]
    train_idx2 = idx[n_valid:]

    fit = train_logit_model(
        X_train=X_train[train_idx2],
        y_train=y_train[train_idx2],
        X_valid=X_train[valid_idx],
        y_valid=y_train[valid_idx],
        lr=1e-3,
        n_epochs=300,
        seed=seed,
        device=device,
        patience=60,
    )

    preds = np.empty(len(grid_pred), dtype=np.float32)

    for start in range(0, len(grid_pred), batch_size_pred):
        end = min(start + batch_size_pred, len(grid_pred))
        chunk = grid_pred.iloc[start:end].copy()

        for c in x_cols:
            chunk[c] = (chunk[c].astype(float) - means[c]) / stds[c]

        X_chunk = chunk[x_cols].to_numpy(dtype=np.float32)
        preds[start:end] = predict_occurrence_probability(fit["model"], X_chunk).astype(np.float32)

        print(f"Occurrence prediction: {end:,}/{len(grid_pred):,}")

    return preds
