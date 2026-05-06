import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import BSpline

from downscaling.egpd import (
    egpd_left_censored_nll,
    egpd_left_censored_nll_sum,
)

from downscaling.config import CENSOR_THRESHOLD, XI_INIT, SIGMA_INIT, KAPPA_INIT, ALLOWED_VARIANTS


def check_variant(variant: str):
    if variant not in ALLOWED_VARIANTS:
        raise ValueError(f"variant must be one of {ALLOWED_VARIANTS}, got {variant}")


def make_bspline_design(x, n_inner_knots=4, degree=3):
    x = np.asarray(x, dtype=float).reshape(-1)

    xmin, xmax = np.min(x), np.max(x)

    if np.isclose(xmin, xmax):
        return np.ones((len(x), 1), dtype=float), {
            "knots": None,
            "degree": degree,
        }

    inner = np.linspace(xmin, xmax, n_inner_knots + 2)[1:-1]

    knots = np.concatenate([
        np.repeat(xmin, degree + 1),
        inner,
        np.repeat(xmax, degree + 1),
    ])

    n_basis = len(knots) - degree - 1
    B = np.zeros((len(x), n_basis), dtype=float)

    for j in range(n_basis):
        coef = np.zeros(n_basis)
        coef[j] = 1.0

        spl = BSpline(knots, coef, degree, extrapolate=True)
        B[:, j] = spl(x)

    return B, {
        "knots": knots,
        "degree": degree,
    }


def transform_params_from_theta(theta, X_sigma, X_kappa, variant, xi_fixed):
    n = X_sigma.shape[0] if X_sigma is not None else X_kappa.shape[0]
    i = 0

    if variant in {"both", "sigma_only"}:
        p_s = X_sigma.shape[1]
        beta_s = theta[i:i + p_s]
        i += p_s

        log_sigma = X_sigma @ beta_s
        sigma = np.exp(log_sigma)

    else:
        log_sigma0 = theta[i]
        i += 1

        sigma = np.exp(log_sigma0) * np.ones(n)

    if variant in {"both", "kappa_only"}:
        p_k = X_kappa.shape[1]
        beta_k = theta[i:i + p_k]
        i += p_k

        log_kappa = X_kappa @ beta_k
        kappa = np.exp(log_kappa)

    else:
        log_kappa0 = theta[i]
        i += 1

        kappa = np.exp(log_kappa0) * np.ones(n)

    xi = np.full(n, xi_fixed, dtype=float)

    return sigma, kappa, xi


def fit_egpd_regression_model(
    y_train,
    x_train,
    model_type="glm",
    variant="both",
    xi_fixed=0.24,
    sigma_init=0.57,
    kappa_init=0.28,
    censor_threshold=0.22,
    lambda_ridge=1e-4,
    lambda_smooth=1e-3,
    n_inner_knots=4,
):
    check_variant(variant)

    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    x_train = np.asarray(x_train, dtype=float).reshape(-1)

    if model_type == "glm":
        x_mean = np.mean(x_train)
        x_sd = np.std(x_train, ddof=1)

        if x_sd <= 0:
            x_sd = 1.0

        z_train = (x_train - x_mean) / x_sd
        X_base_train = np.column_stack([np.ones(len(z_train)), z_train])

        basis_meta = {
            "type": "glm",
            "x_mean": x_mean,
            "x_sd": x_sd,
        }

    elif model_type == "gam":
        X_base_train, bs_meta = make_bspline_design(
            x_train,
            n_inner_knots=n_inner_knots,
            degree=3,
        )

        basis_meta = {
            "type": "gam",
            **bs_meta,
        }

    else:
        raise ValueError("model_type must be 'glm' or 'gam'.")

    X_sigma_train = X_base_train if variant in {"both", "sigma_only"} else None
    X_kappa_train = X_base_train if variant in {"both", "kappa_only"} else None

    if variant == "both":
        theta0 = np.concatenate([
            np.zeros(X_base_train.shape[1]),
            np.zeros(X_base_train.shape[1]),
        ])

        theta0[0] = np.log(sigma_init)
        theta0[X_base_train.shape[1]] = np.log(kappa_init)

    elif variant == "sigma_only":
        theta0 = np.concatenate([
            np.zeros(X_base_train.shape[1]),
            np.array([np.log(kappa_init)]),
        ])

        theta0[0] = np.log(sigma_init)

    elif variant == "kappa_only":
        theta0 = np.concatenate([
            np.array([np.log(sigma_init)]),
            np.zeros(X_base_train.shape[1]),
        ])

        theta0[1] = np.log(kappa_init)

    def objective(theta):
        sigma, kappa, xi = transform_params_from_theta(
            theta=theta,
            X_sigma=X_sigma_train,
            X_kappa=X_kappa_train,
            variant=variant,
            xi_fixed=xi_fixed,
        )

        nll = egpd_left_censored_nll(
            y=y_train,
            sigma=sigma,
            kappa=kappa,
            xi=xi,
            c=censor_threshold,
        )

        pen = lambda_ridge * np.sum(theta ** 2)

        if model_type == "gam":
            if variant == "both":
                p = X_base_train.shape[1]
                beta_s = theta[:p]
                beta_k = theta[p:(2 * p)]

                pen += lambda_smooth * np.sum(np.diff(beta_s, n=2) ** 2)
                pen += lambda_smooth * np.sum(np.diff(beta_k, n=2) ** 2)

            elif variant == "sigma_only":
                p = X_base_train.shape[1]
                beta_s = theta[:p]

                pen += lambda_smooth * np.sum(np.diff(beta_s, n=2) ** 2)

            elif variant == "kappa_only":
                p = X_base_train.shape[1]
                beta_k = theta[1:(1 + p)]

                pen += lambda_smooth * np.sum(np.diff(beta_k, n=2) ** 2)

        return nll + pen

    res = minimize(objective, theta0, method="L-BFGS-B")

    theta_hat = res.x

    sigma_train, kappa_train, xi_train = transform_params_from_theta(
        theta_hat,
        X_sigma_train,
        X_kappa_train,
        variant,
        xi_fixed,
    )

    out = {
        "success": bool(res.success),
        "message": res.message,
        "theta_hat": theta_hat,
        "basis_meta": basis_meta,
        "variant": variant,
        "model_type": model_type,
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

    return out


def predict_egpd_regression_model(fit, x_new, xi_fixed):
    x_new = np.asarray(x_new, dtype=float).reshape(-1)

    model_type = fit["model_type"]
    variant = fit["variant"]
    theta_hat = fit["theta_hat"]
    basis_meta = fit["basis_meta"]

    if model_type == "glm":
        z = (x_new - basis_meta["x_mean"]) / basis_meta["x_sd"]
        X_base = np.column_stack([np.ones(len(z)), z])

    else:
        if basis_meta["knots"] is None:
            X_base = np.ones((len(x_new), 1), dtype=float)

        else:
            knots = basis_meta["knots"]
            degree = basis_meta["degree"]
            n_basis = len(knots) - degree - 1

            X_base = np.zeros((len(x_new), n_basis), dtype=float)

            for j in range(n_basis):
                coef = np.zeros(n_basis)
                coef[j] = 1.0

                spl = BSpline(knots, coef, degree, extrapolate=True)
                X_base[:, j] = spl(x_new)

    X_sigma = X_base if variant in {"both", "sigma_only"} else None
    X_kappa = X_base if variant in {"both", "kappa_only"} else None

    sigma, kappa, xi = transform_params_from_theta(
        theta_hat,
        X_sigma,
        X_kappa,
        variant,
        xi_fixed=xi_fixed,
    )

    return {
        "pred_sigma": sigma,
        "pred_kappa": kappa,
        "pred_xi": xi,
    }


def get_gam_initial_values(
    df_train: pd.DataFrame,
    covariate_col: str,
    variant: str = "both",
    xi_fixed: float = XI_INIT,
    sigma_init: float = SIGMA_INIT,
    kappa_init: float = KAPPA_INIT,
    censor_threshold: float = CENSOR_THRESHOLD,
) -> dict:
    """
    Fit a one-covariate GAM on the training data and use the median fitted
    sigma/kappa as scalar NN initial values.

    This does not initialize NN weights with the GAM function.
    It initializes the distribution level around a meaningful GAM solution.
    """
    y_train = df_train["Y_obs"].to_numpy(float)
    x_train = df_train[covariate_col].to_numpy(float)

    fit_gam = fit_egpd_regression_model(
        y_train=y_train,
        x_train=x_train,
        model_type="gam",
        variant=variant,
        xi_fixed=xi_fixed,
        sigma_init=sigma_init,
        kappa_init=kappa_init,
        censor_threshold=censor_threshold,
    )

    pred_train = predict_egpd_regression_model(
        fit_gam,
        x_train,
        xi_fixed=xi_fixed,
    )

    sigma_med = float(np.nanmedian(pred_train["pred_sigma"]))
    kappa_med = float(np.nanmedian(pred_train["pred_kappa"]))

    sigma_med = float(np.clip(sigma_med, 1e-3, 100.0))
    kappa_med = float(np.clip(kappa_med, 5e-2, 50.0))

    return {
        "sigma_init_gam": sigma_med,
        "kappa_init_gam": kappa_med,
        "gam_success": bool(fit_gam["success"]),
        "gam_train_nll": float(fit_gam["train_nll"]),
    }
