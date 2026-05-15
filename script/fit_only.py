# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from downscaling.config import SIGMA_INIT, KAPPA_INIT, XI_INIT
from downscaling.paths import DOWNSCALING_TABLE
from downscaling.data import prepare_modeling_dataframe
from downscaling.stationary import fit_egpd_stationary_direct


# %%
# Load data
df_raw = pd.read_csv(DOWNSCALING_TABLE, sep=";")
df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True)

df_model, x_cols27, x_cols_dt0h, x_cols_all = prepare_modeling_dataframe(df_raw)

y = df_model["Y_obs"].to_numpy(float)
y = y[np.isfinite(y)]

print("n =", len(y))
print("zero proportion =", np.mean(y == 0))
print("min/max =", np.min(y), np.max(y))


# %%
# Fit simple stationary EGPD on all Y_obs
# No censoring: censor_threshold=0.0
fit = fit_egpd_stationary_direct(
    y_train=y,
    y_valid=y,
    sigma_init=SIGMA_INIT,
    kappa_init=KAPPA_INIT,
    xi_init=XI_INIT,
    fix_xi=False,
    censor_threshold=0.0,
)

print(fit)


sigma_hat = float(fit["sigma_hat"])
kappa_hat = float(fit["kappa_hat"])
xi_hat = float(fit["xi_hat"])

print("sigma_hat =", sigma_hat)
print("kappa_hat =", kappa_hat)
print("xi_hat    =", xi_hat)


# %%
# EGPD quantile function

def egpd_ppf(p, sigma, kappa, xi, eps=1e-12):
    p = np.clip(np.asarray(p), eps, 1.0 - eps)

    u = p ** (1.0 / kappa)

    if abs(xi) > eps:
        q = sigma / xi * ((1.0 - u) ** (-xi) - 1.0)
    else:
        q = -sigma * np.log(1.0 - u)

    return q


# %%
# Classical QQ-plot: theoretical EGPD quantiles vs empirical quantiles

y_sorted = np.sort(y)
n = len(y_sorted)

p = (np.arange(1, n + 1) - 0.5) / n
q_theo = egpd_ppf(p, sigma_hat, kappa_hat, xi_hat)

plt.figure(figsize=(6, 6))
plt.scatter(q_theo, y_sorted, s=8, alpha=0.5)

lim_max = max(np.nanmax(q_theo), np.nanmax(y_sorted))
plt.plot([0, lim_max], [0, lim_max], linestyle="--")

plt.xlabel("Theoretical EGPD quantiles")
plt.ylabel("Empirical quantiles of Y_obs")
plt.title("Stationary EGPD QQ-plot on Y_obs")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# Optional: focus on the upper tail only

p_low = 0.50
p_high = 0.999

keep = (p >= p_low) & (p <= p_high)

plt.figure(figsize=(6, 6))
plt.scatter(q_theo[keep], y_sorted[keep], s=10, alpha=0.6)

lim_max = max(np.nanmax(q_theo[keep]), np.nanmax(y_sorted[keep]))
plt.plot([0, lim_max], [0, lim_max], linestyle="--")

plt.xlabel("Theoretical EGPD quantiles")
plt.ylabel("Empirical quantiles of Y_obs")
plt.title(f"Stationary EGPD QQ-plot on Y_obs, p ∈ [{p_low}, {p_high}]")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()