from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import math
import numpy as np
import torch
import torch.nn as nn


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


def egpd_left_censored_nll_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    c: float = 0.22,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Left-censored negative log-likelihood for EGPD.

    For y > c, use log-density.
    For y <= c, use log CDF at c.
    """
    sigma = y_pred[..., 0].clamp(min=1e-8, max=1e3)
    kappa = y_pred[..., 1].clamp(min=1e-8, max=1e3)
    xi = y_pred[..., 2].clamp(min=1e-8, max=1e3)

    is_obs = (y_true >= 0).to(y_true.dtype)
    y = y_true

    c_t = torch.as_tensor(c, dtype=y.dtype, device=y.device)

    unc = ((y > c_t).to(y.dtype)) * is_obs
    cen = ((y <= c_t).to(y.dtype)) * is_obs

    z = (1.0 + xi * y / sigma).clamp_min(1e-10)
    t = 1.0 - z.pow(-1.0 / xi)
    t = t.clamp(min=eps, max=1.0 - eps)

    logf = (
        torch.log(kappa)
        - torch.log(sigma)
        - (1.0 / xi + 1.0) * torch.log(z)
        + (kappa - 1.0) * torch.log(t)
    )

    zc = (1.0 + xi * c_t / sigma).clamp_min(1e-10)
    tc = 1.0 - zc.pow(-1.0 / xi)
    tc = tc.clamp(min=eps, max=1.0 - eps)

    Fc = tc.pow(kappa).clamp(min=eps, max=1.0)
    logFc = torch.log(Fc)

    ll = unc * logf + cen * logFc

    denom = is_obs.sum().clamp_min(1.0)
    nll = -(ll.sum() / denom)

    if not torch.isfinite(nll):
        return torch.tensor(1e6, device=y_true.device, dtype=y_true.dtype)

    return nll


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
    clipnorm=1.0,
    seed=1,
    device=None,
    dtype=torch.float32,
    early_stopping=True,
    patience=20,
    min_delta=0.0,
    warmup_epochs=0,
    restore_best=True,
    censor_threshold=0.22,
):
    import copy
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(dev)

    def to_tensor_or_none(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.to(device=dev, dtype=dtype)
        return torch.as_tensor(x, dtype=dtype, device=dev)

    Xs = to_tensor_or_none(X_s)
    Xk = to_tensor_or_none(X_k)
    Off = to_tensor_or_none(offset)
    Yt = to_tensor_or_none(Y_train)

    Xs_valid_t = to_tensor_or_none(X_s_valid)
    Xk_valid_t = to_tensor_or_none(X_k_valid)
    Off_valid_t = to_tensor_or_none(offset_valid)
    Yv = to_tensor_or_none(Y_valid)

    n_train = Yt.shape[0]
    idx_all = torch.arange(n_train, device=dev)

    train_loader = DataLoader(
        TensorDataset(idx_all),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_epoch_losses = []
    val_epoch_losses = []

    best_score = float("inf")
    best_state = None
    bad_epochs = 0
    stopped_epoch = n_epochs

    def subset_or_none(x, idx):
        if x is None:
            return None
        return x[idx]

    def make_inputs(Xs_part, Xk_part, offset_part):
        return EGPDNNOnlyInputs(
            X_s=Xs_part,
            X_k=Xk_part,
            offset=offset_part,
        )

    for ep in range(1, n_epochs + 1):
        model.train()
        batch_losses = []

        for (idx_batch,) in train_loader:
            Xs_b = subset_or_none(Xs, idx_batch)
            Xk_b = subset_or_none(Xk, idx_batch)
            Off_b = subset_or_none(Off, idx_batch)
            Y_b = Yt[idx_batch]

            pred_b = model(make_inputs(Xs_b, Xk_b, Off_b))
            loss_b = egpd_left_censored_nll_loss(
                y_true=Y_b,
                y_pred=pred_b,
                c=censor_threshold,
            )

            opt.zero_grad()
            loss_b.backward()

            if clipnorm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)

            opt.step()
            batch_losses.append(float(loss_b.detach().cpu()))

        train_loss = float(np.mean(batch_losses))
        train_epoch_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            if Yv is not None:
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
                score = train_loss

        use_early_stopping = early_stopping and (ep > warmup_epochs)

        improved = (best_score - score) > min_delta
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if use_early_stopping and (Yv is not None) and (bad_epochs >= patience):
            stopped_epoch = ep
            break

    if restore_best and (best_state is not None):
        model.load_state_dict(best_state)

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