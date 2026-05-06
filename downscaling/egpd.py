
import numpy as np
from scipy.special import beta as beta_function


def qegpd(prob, sigma, kappa, xi, eps=1e-12, kappa_min=5e-2, xi_min=1e-6):
    """
    Quantile function of the EGPD distribution.
    """
    prob = np.asarray(prob, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), eps)
    kappa = np.maximum(np.asarray(kappa, dtype=float), kappa_min)
    xi = np.maximum(np.asarray(xi, dtype=float), xi_min)

    prob = np.clip(prob, eps, 1.0 - eps)
    p1k = np.exp(np.log(prob) / kappa)
    t = np.maximum(1.0 - p1k, eps)

    return sigma / xi * (np.power(t, -xi) - 1.0)


def egpd_cdf(y, sigma, kappa, xi, eps=1e-12, kappa_min=5e-2, xi_min=1e-6):
    """
    CDF of the EGPD distribution.
    F(y) = G(y)^kappa where G is the GPD CDF.
    """
    y = np.asarray(y, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), eps)
    kappa = np.maximum(np.asarray(kappa, dtype=float), kappa_min)
    xi = np.maximum(np.asarray(xi, dtype=float), xi_min)

    t = 1.0 + xi * y / sigma
    t = np.maximum(t, eps)

    a = np.power(t, -1.0 / xi)
    inner = np.clip(1.0 - a, eps, 1.0 - eps)

    return np.clip(np.power(inner, kappa), eps, 1.0 - eps)


def egpd_logpdf(y, sigma, kappa, xi, eps=1e-12):
    """
    Log-PDF of the EGPD distribution.
    """
    y = np.asarray(y, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), eps)
    kappa = np.maximum(np.asarray(kappa, dtype=float), eps)
    xi = np.maximum(np.asarray(xi, dtype=float), eps)

    z = np.maximum(1.0 + xi * y / sigma, eps)
    t = 1.0 - np.power(z, -1.0 / xi)
    t = np.clip(t, eps, 1.0 - eps)

    return (
        np.log(kappa)
        - np.log(sigma)
        - (1.0 / xi + 1.0) * np.log(z)
        + (kappa - 1.0) * np.log(t)
    )


def egpd_left_censored_loglik(y, sigma, kappa, xi, c=0.22, eps=1e-12):
    """Log-likelihood of the left-censored EGPD distribution.
    For y <= c, contribution is log(F(c))."""
    y = np.asarray(y, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    kappa = np.asarray(kappa, dtype=float).reshape(-1)
    xi = np.asarray(xi, dtype=float).reshape(-1)

    unc = y > c # uncensored
    cen = ~unc # left-censored

    ll = np.zeros_like(y, dtype=float)

    # Uncensored contribution with log PDF
    if np.any(unc):
        ll[unc] = egpd_logpdf(
            y[unc],
            sigma[unc],
            kappa[unc],
            xi[unc],
            eps=eps,
        )

    # Censored contribution with log CDF at c
    if np.any(cen):
        Fc = egpd_cdf(
            np.full(np.sum(cen), c),
            sigma[cen],
            kappa[cen],
            xi[cen],
            eps=eps,
        )
        ll[cen] = np.log(np.clip(Fc, eps, 1.0))

    return ll


def egpd_left_censored_nll(y, sigma, kappa, xi, c=0.22, eps=1e-12):
    """
    Negative log-likelihood of the left-censored EGPD distribution, averaged over observations.
    For y <= c, contribution is log(F(c))."""
    ll = egpd_left_censored_loglik(
        y,
        sigma,
        kappa,
        xi,
        c=c,
        eps=eps,
    )
    nll = -np.mean(ll)

    if not np.isfinite(nll):
        return 1e6

    return float(nll)


def egpd_left_censored_nll_sum(y, sigma, kappa, xi, c=0.22, eps=1e-12,):
    """Negative log-likelihood of the left-censored EGPD distribution, summed over observations."""
    ll = egpd_left_censored_loglik(
        y,
        sigma,
        kappa,
        xi,
        c=c,
        eps=eps,
    )
    nll = -np.sum(ll)

    if not np.isfinite(nll):
        return 1e12

    return float(nll)



def compute_pit(y, sigma, kappa, xi):
    """
    Compute the Probability Integral Transform (PIT) values for observations y given EGPD parameters.
    """
    return egpd_cdf(y, sigma, kappa, xi)


def predicted_mean_mc(sigma, kappa, xi, n_mc=1000, seed=123):
    """
    Compute the predicted mean of the EGPD distribution using Monte Carlo sampling.
    """
    rng = np.random.default_rng(seed)

    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    kappa = np.asarray(kappa, dtype=float).reshape(-1)
    xi = np.asarray(xi, dtype=float).reshape(-1)

    u = rng.uniform(size=(n_mc, len(sigma)))

    samples = qegpd(
        u,
        sigma[None, :],
        kappa[None, :],
        xi[None, :],
    )

    return samples.mean(axis=0)


def egpd_mean(sigma, kappa, xi, eps=1e-12):
    """
    Mean of the EGPD.
    If F(y) = G(y)^kappa with G a GPD CDF, then:
    E[Y] = sigma / xi * (kappa * B(kappa, 1 - xi) - 1).
    """
    sigma = np.asarray(sigma, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    xi = np.asarray(xi, dtype=float)

    out = np.full_like(sigma, np.nan, dtype=float)
    ok = xi < 1.0 # ok for xi < 1 to have finite mean

    out[ok] = (
        sigma[ok] / np.maximum(xi[ok], eps)
        * (
            kappa[ok] * beta_function(kappa[ok], 1.0 - xi[ok])
            - 1.0
        )
    )

    return out
