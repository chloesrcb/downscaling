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


