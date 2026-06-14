from __future__ import annotations

import ast
import copy
import inspect
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from downscaling.data import (
    build_X_from_meta,
    build_xy_train_valid,
    standardize_train_only,
)
from downscaling.settings import ALLOWED_VARIANTS

@dataclass
class EGPDNNOnlyInputs:
    X_s: Optional[torch.Tensor] = None
    X_k: Optional[torch.Tensor] = None
    offset: Optional[torch.Tensor] = None


class MLP(nn.Module):
    def __init__(self, in_dim: int, widths: Sequence[int], out_dim: int = 1):
        super().__init__()

        layers = []
        prev = in_dim
        for w in widths:
            layers.append(nn.Linear(prev, w))
            layers.append(nn.ReLU())
            prev = w
        layers.append(nn.Linear(prev, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EGPDPINN_NNOnly(nn.Module):
    """
    Neural network EGPD model with optional predictor branches for sigma and kappa.

    sigma(x) = exp(f_s(x)) * offset if offset is provided
    kappa(x) = exp(f_k(x))
    xi is fixed and stationary
    """
    def __init__(
        self,
        d_s: Optional[int],
        d_k: Optional[int],
        init_scale: float,
        init_kappa: float,
        init_xi: float,
        widths: Sequence[int] = (16, 8),
    ):
        super().__init__()

        if d_s is None and d_k is None:
            raise ValueError("At least one of d_s or d_k must be provided.")

        if init_scale <= 0 or init_kappa <= 0:
            raise ValueError("init_scale and init_kappa must be > 0.")

        if not (0.0 < init_xi < 1.0):
            raise ValueError("init_xi must be in (0, 1).")

        self.d_s = d_s
        self.d_k = d_k

        init_log_sigma = math.log(init_scale)
        init_log_kappa = math.log(init_kappa)

        if d_s is not None:
            self.nn_s = MLP(d_s, widths, out_dim=1)
            last = self.nn_s.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, init_log_sigma)
        else:
            self.nn_s = None
            self.log_sigma_const = nn.Parameter(
                torch.tensor(init_log_sigma, dtype=torch.float32),
                requires_grad=False
            )

        if d_k is not None:
            self.nn_k = MLP(d_k, widths, out_dim=1)
            last = self.nn_k.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, init_log_kappa)
        else:
            self.nn_k = None
            self.log_kappa_const = nn.Parameter(
                torch.tensor(init_log_kappa, dtype=torch.float32),
                requires_grad=False
            )

        xi_logit = math.log(init_xi / (1.0 - init_xi))
        self._xi_logit = nn.Parameter(
            torch.tensor(xi_logit, dtype=torch.float32),
            requires_grad=False
        )

    def forward(self, inp: EGPDNNOnlyInputs) -> torch.Tensor:
        base = inp.X_s if inp.X_s is not None else inp.X_k
        if base is None:
            raise ValueError("At least one of X_s or X_k must be provided.")

        batch_shape = base.shape[:-1]
        device = base.device
        dtype = base.dtype

        if self.nn_s is None:
            log_sigma = self.log_sigma_const.to(device=device, dtype=dtype).expand(*batch_shape, 1)
        else:
            if inp.X_s is None:
                raise ValueError("X_s is required because d_s was provided.")
            log_sigma = self.nn_s(inp.X_s)

        if self.nn_k is None:
            log_kappa = self.log_kappa_const.to(device=device, dtype=dtype).expand(*batch_shape, 1)
        else:
            if inp.X_k is None:
                raise ValueError("X_k is required because d_k was provided.")
            log_kappa = self.nn_k(inp.X_k)

        sigma = torch.exp(log_sigma).clamp(min=1e-8, max=1e3)
        kappa = torch.exp(log_kappa).clamp(min=1e-8, max=1e3)

        if inp.offset is not None:
            off = inp.offset
            if off.ndim == len(batch_shape):
                off = off.unsqueeze(-1)
            sigma = sigma * off

        xi = torch.sigmoid(self._xi_logit.to(device=device, dtype=dtype)).expand(*batch_shape, 1)
        xi = xi.clamp(min=1e-8, max=1 - 1e-8)

        return torch.cat([sigma, kappa, xi], dim=-1)




def egpd_cdf_torch(y, sigma, kappa, xi, eps=1e-12):
    """
    Torch EGPD CDF.
    """
    sigma = torch.clamp(sigma, min=eps)
    kappa = torch.clamp(kappa, min=eps)
    xi = torch.clamp(xi, min=eps)

    t = 1.0 + xi * y / sigma
    t = torch.clamp(t, min=eps)

    a = torch.pow(t, -1.0 / xi)
    inner = torch.clamp(1.0 - a, min=eps, max=1.0 - eps)

    return torch.clamp(torch.pow(inner, kappa), min=eps, max=1.0 - eps)


def egpd_logpdf_torch(y, sigma, kappa, xi, eps=1e-12):
    """
    Torch EGPD log-density.
    """
    sigma = torch.clamp(sigma, min=eps)
    kappa = torch.clamp(kappa, min=eps)
    xi = torch.clamp(xi, min=eps)

    z = torch.clamp(1.0 + xi * y / sigma, min=eps)
    t = torch.clamp(1.0 - torch.pow(z, -1.0 / xi), min=eps, max=1.0 - eps)

    logf = (
        torch.log(kappa)
        - torch.log(sigma)
        - (1.0 / xi + 1.0) * torch.log(z)
        + (kappa - 1.0) * torch.log(t)
    )

    return logf


def egpd_left_censored_nll_torch(
    y,
    sigma,
    kappa,
    xi,
    censor_threshold=0.22,
    eps=1e-12,
):
    """
    Mean negative log-likelihood for left-censored EGPD.

    For y <= censor_threshold, contribution is log(F(c)).
    For y > censor_threshold, contribution is log(f(y)).
    """
    y = y.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    kappa = kappa.reshape(-1, 1)
    xi = xi.reshape(-1, 1)

    uncensored = y > censor_threshold
    censored = ~uncensored

    ll = torch.zeros_like(y)

    if torch.any(uncensored):
        ll[uncensored] = egpd_logpdf_torch(
            y[uncensored],
            sigma[uncensored],
            kappa[uncensored],
            xi[uncensored],
            eps=eps,
        )

    if torch.any(censored):
        c = torch.full_like(y[censored], float(censor_threshold))

        Fc = egpd_cdf_torch(
            c,
            sigma[censored],
            kappa[censored],
            xi[censored],
            eps=eps,
        )

        ll[censored] = torch.log(torch.clamp(Fc, min=eps, max=1.0))

    return -torch.mean(ll)





def egpd_left_censored_nll_loss(y_true, y_pred, c: float = 0.22):
    return egpd_left_censored_nll_torch(
        y=y_true,
        sigma=y_pred[..., 0],
        kappa=y_pred[..., 1],
        xi=y_pred[..., 2],
        censor_threshold=c,
    )


def train_egpd_nn_only(
    model,
    X_s,
    X_k,
    Y_train,
    offset=None,
    X_s_valid=None,
    X_k_valid=None,
    offset_valid=None,
    Y_valid=None,
    n_epochs=100,
    batch_size=64,
    lr=1e-3,
    weight_decay=0.0,
    clipnorm=1.0,
    seed=1,
    device=None,
    dtype=torch.float32,
    early_stopping=True,
    patience=25,
    min_delta=0.0,
    warmup_epochs=20,
    restore_best=True,
    censor_threshold=0.22,
    kappa_max_nn=1.5,
    lambda_kappa=0.05,
    ):
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine device
    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(dev)

    # Function to convert inputs to tensors or return None
    def to_tensor_or_none(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.to(device=dev, dtype=dtype)
        return torch.as_tensor(x, dtype=dtype, device=dev)

    # Convert all inputs to tensors
    Xs = to_tensor_or_none(X_s)
    Xk = to_tensor_or_none(X_k)
    Off = to_tensor_or_none(offset)
    Yt = to_tensor_or_none(Y_train)

    Xs_valid_t = to_tensor_or_none(X_s_valid)
    Xk_valid_t = to_tensor_or_none(X_k_valid)
    Off_valid_t = to_tensor_or_none(offset_valid)
    Yv = to_tensor_or_none(Y_valid)

    # Create DataLoader for training data
    n_train = Yt.shape[0]
    idx_all = torch.arange(n_train, device=dev)

    train_loader = DataLoader(
        TensorDataset(idx_all),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    # Initialize optimizer: Adam
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Inits
    train_epoch_losses = []
    val_epoch_losses = []

    best_score = float("inf")
    best_state = None
    best_epoch = None
    bad_epochs = 0
    stopped_epoch = n_epochs

    # function to subset tensors or return None
    def subset_or_none(x, idx):
        if x is None:
            return None
        return x[idx]

    # function to create model inputs
    def make_inputs(Xs_part, Xk_part, offset_part):
        return EGPDNNOnlyInputs(
            X_s=Xs_part,
            X_k=Xk_part,
            offset=offset_part,
        )

    # Training loop
    for ep in range(1, n_epochs + 1):
        model.train() # Set model to training mode
        batch_losses = [] # List to store batch losses for each epoch

        # Iterate over batches
        for (idx_batch,) in train_loader:
            # for each batch, subset the inputs
            Xs_b = subset_or_none(Xs, idx_batch)
            Xk_b = subset_or_none(Xk, idx_batch)
            Off_b = subset_or_none(Off, idx_batch) # Offset
            Y_b = Yt[idx_batch] # Target variable for the batch
            
            # Forward pass
            pred_b = model(make_inputs(Xs_b, Xk_b, Off_b))
            # Compute loss
            loss_b = egpd_left_censored_nll_loss(
                y_true=Y_b,
                y_pred=pred_b,
                c=censor_threshold,
            )

            # kappa penalty
            pred_kappa = pred_b[..., 1]

            kappa_penalty = torch.mean(
                torch.relu(pred_kappa - kappa_max_nn) ** 2
            )

            loss_b = loss_b + lambda_kappa * kappa_penalty
            
            # Backward pass and optimization step
            opt.zero_grad()
            loss_b.backward()

            # Gradient clipping
            if clipnorm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)

            # Update parameters
            opt.step()
            # Store batch loss
            batch_losses.append(float(loss_b.detach().cpu()))

        # Compute average training loss for the epoch
        train_loss = float(np.mean(batch_losses))
        train_epoch_losses.append(train_loss)

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            if Yv is not None:
                # Compute validation loss, if validation data is provided
                pred_val = model(make_inputs(Xs_valid_t, Xk_valid_t, Off_valid_t))
                val_loss = float(
                    egpd_left_censored_nll_loss(
                        y_true=Yv,
                        y_pred=pred_val,
                        c=censor_threshold,
                    ).detach().cpu()
                )
                val_epoch_losses.append(val_loss)
                score = val_loss
            else:
                # If no validation data, use training loss for early stopping
                score = train_loss

        # Determine if we should use early stopping
        use_early_stopping = early_stopping and (ep > warmup_epochs)
        
        # Check for improvement and update best score/state
        improved = (best_score - score) > min_delta
        if ep <= warmup_epochs:
            if improved:
                best_score = score
                best_epoch = ep
                best_state = copy.deepcopy(model.state_dict())
            continue
        if improved:
            best_score = score
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if use_early_stopping and (Yv is not None) and (bad_epochs >= patience):
            stopped_epoch = ep
            break

    # Restore best model state if requested
    if restore_best and (best_state is not None):
        model.load_state_dict(best_state)

    # Final evaluation on training (and validation if available)
    model.eval()
    with torch.no_grad():
        pred_train = model(make_inputs(Xs, Xk, Off))
        final_train = float(
            egpd_left_censored_nll_loss(
                y_true=Yt,
                y_pred=pred_train,
                c=censor_threshold,
            ).detach().cpu()
        )

        out = {
            "model": model,
            "train_nll": final_train,
            "history": {
                "train_epoch": train_epoch_losses,
                "val_epoch": val_epoch_losses,
            },
            "best_score": float(best_score),
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "stopped_epoch": int(stopped_epoch),
        }

        if Yv is not None:
            pred_val = model(make_inputs(Xs_valid_t, Xk_valid_t, Off_valid_t))
            final_val = float(
                egpd_left_censored_nll_loss(
                    y_true=Yv,
                    y_pred=pred_val,
                    c=censor_threshold,
                ).detach().cpu()
            )
            out["val_nll"] = final_val

    return out


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
    min_delta: float = 0,
    kappa_max_nn: float = 1.5,
    lambda_kappa: float = 0.05,
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
        min_delta=min_delta,
        warmup_epochs=warmup_epochs,
        censor_threshold=censor_threshold,
        kappa_max_nn=kappa_max_nn,
        lambda_kappa=lambda_kappa,
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
