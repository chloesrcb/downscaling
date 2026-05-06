
import torch


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

# def egpd_left_censored_nll_loss(
#     y_true: torch.Tensor,
#     y_pred: torch.Tensor,
#     c: float = 0.22,
#     eps: float = 1e-12
# ) -> torch.Tensor:
#     """
#     Left-censored negative log-likelihood for EGPD.

#     For y > c, use log-density.
#     For y <= c, use log CDF at c.
#     """
#     sigma = y_pred[..., 0].clamp(min=1e-8, max=1e3)
#     kappa = y_pred[..., 1].clamp(min=1e-8, max=1e3)
#     xi = y_pred[..., 2].clamp(min=1e-8, max=1e3)

#     is_obs = (y_true >= 0).to(y_true.dtype)
#     y = y_true

#     c_t = torch.as_tensor(c, dtype=y.dtype, device=y.device)

#     unc = ((y > c_t).to(y.dtype)) * is_obs
#     cen = ((y <= c_t).to(y.dtype)) * is_obs

#     z = (1.0 + xi * y / sigma).clamp_min(1e-10)
#     t = 1.0 - z.pow(-1.0 / xi)
#     t = t.clamp(min=eps, max=1.0 - eps)

#     logf = (
#         torch.log(kappa)
#         - torch.log(sigma)
#         - (1.0 / xi + 1.0) * torch.log(z)
#         + (kappa - 1.0) * torch.log(t)
#     )

#     zc = (1.0 + xi * c_t / sigma).clamp_min(1e-10)
#     tc = 1.0 - zc.pow(-1.0 / xi)
#     tc = tc.clamp(min=eps, max=1.0 - eps)

#     Fc = tc.pow(kappa).clamp(min=eps, max=1.0)
#     logFc = torch.log(Fc)

#     ll = unc * logf + cen * logFc

#     denom = is_obs.sum().clamp_min(1.0)
#     nll = -(ll.sum() / denom)

#     if not torch.isfinite(nll):
#         return torch.tensor(1e6, device=y_true.device, dtype=y_true.dtype)

#     return nll


