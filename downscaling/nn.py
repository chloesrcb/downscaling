import ast
import inspect
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

from downscaling.models import EGPDPINN_NNOnly
from downscaling.training import train_egpd_nn_only
from downscaling.config import ALLOWED_VARIANTS
from downscaling.features import (
    standardize_train_only,
    build_xy_train_valid,
    build_X_from_meta,
)
from downscaling.scores import summarize_distribution_metrics
from downscaling.egpd import egpd_left_censored_nll_sum


@dataclass
class Config:
    name: str
    widths: Tuple[int, ...] = (16, 8)


def parse_widths(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    if isinstance(x, np.ndarray):
        return tuple(x.tolist())
    if isinstance(x, str):
        return tuple(ast.literal_eval(x))
    return tuple(x)


def check_variant(variant: str):
    if variant not in ALLOWED_VARIANTS:
        raise ValueError(f"variant must be one of {ALLOWED_VARIANTS}, got {variant}")


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
    from downscaling.prediction import predict_params

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
