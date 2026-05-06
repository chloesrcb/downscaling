
import math
import numpy as np
from scipy.optimize import minimize

from downscaling.egpd import (
    egpd_left_censored_nll,
    egpd_left_censored_nll_sum,
)

from downscaling.config import SIGMA_INIT, KAPPA_INIT, XI_INIT


def fit_egpd_stationary_direct(
    y_train,
    y_valid=None,
    sigma_init=SIGMA_INIT,
    kappa_init=KAPPA_INIT,
    xi_init=XI_INIT,
    fix_xi=True,
    censor_threshold=0.22,
):
    """
    Fit a stationary EGPD model.
    """
    y_train = np.asarray(y_train, dtype=float).reshape(-1)

    if fix_xi:
        theta0 = np.array([np.log(sigma_init), np.log(kappa_init)], dtype=float)

        def objective(theta):
            log_sigma, log_kappa = theta

            sigma = np.exp(log_sigma) * np.ones_like(y_train)
            kappa = np.exp(log_kappa) * np.ones_like(y_train)
            xi = np.full_like(y_train, xi_init, dtype=float)

            return egpd_left_censored_nll(
                y=y_train,
                sigma=sigma,
                kappa=kappa,
                xi=xi,
                c=censor_threshold,
            )

        res = minimize(objective, theta0, method="L-BFGS-B")

        log_sigma_hat, log_kappa_hat = res.x

        sigma_hat = float(np.exp(log_sigma_hat))
        kappa_hat = float(np.exp(log_kappa_hat))
        xi_hat = float(xi_init)

    else:
        theta0 = np.array(
            [
                np.log(sigma_init),
                np.log(kappa_init),
                math.log(xi_init / (1.0 - xi_init)),
            ],
            dtype=float,
        )

        def objective(theta):
            log_sigma, log_kappa, xi_logit = theta

            sigma = np.exp(log_sigma) * np.ones_like(y_train)
            kappa = np.exp(log_kappa) * np.ones_like(y_train)
            xi_scalar = 1.0 / (1.0 + np.exp(-xi_logit))
            xi = np.full_like(y_train, xi_scalar, dtype=float)

            return egpd_left_censored_nll(
                y=y_train,
                sigma=sigma,
                kappa=kappa,
                xi=xi,
                c=censor_threshold,
            )

        res = minimize(objective, theta0, method="L-BFGS-B")

        log_sigma_hat, log_kappa_hat, xi_logit_hat = res.x

        sigma_hat = float(np.exp(log_sigma_hat))
        kappa_hat = float(np.exp(log_kappa_hat))
        xi_hat = float(1.0 / (1.0 + np.exp(-xi_logit_hat)))

    sigma_train = np.full_like(y_train, sigma_hat, dtype=float)
    kappa_train = np.full_like(y_train, kappa_hat, dtype=float)
    xi_train = np.full_like(y_train, xi_hat, dtype=float)

    out = {
        "success": bool(res.success),
        "message": res.message,
        "sigma_hat": sigma_hat,
        "kappa_hat": kappa_hat,
        "xi_hat": xi_hat,
        "train_nll": egpd_left_censored_nll(
            y_train,
            sigma_train,
            kappa_train,
            xi_train,
            c=censor_threshold,
        ),
        "train_nll_sum": egpd_left_censored_nll_sum(
            y_train,
            sigma_train,
            kappa_train,
            xi_train,
            c=censor_threshold,
        ),
    }

    if y_valid is not None:
        y_valid = np.asarray(y_valid, dtype=float).reshape(-1)

        sigma_valid = np.full_like(y_valid, sigma_hat, dtype=float)
        kappa_valid = np.full_like(y_valid, kappa_hat, dtype=float)
        xi_valid = np.full_like(y_valid, xi_hat, dtype=float)

        out["val_nll"] = egpd_left_censored_nll(
            y_valid,
            sigma_valid,
            kappa_valid,
            xi_valid,
            c=censor_threshold,
        )
        out["val_nll_sum"] = egpd_left_censored_nll_sum(
            y_valid,
            sigma_valid,
            kappa_valid,
            xi_valid,
            c=censor_threshold,
        )

    return out