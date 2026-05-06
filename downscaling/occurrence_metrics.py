
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Numerical helpers for logistic predictions and metrics
def sigmoid_np(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def brier_score_manual(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    p_pred = np.asarray(p_pred, dtype=float).reshape(-1)
    return float(np.mean((p_pred - y_true) ** 2))


def accuracy_manual(y_true: np.ndarray, p_pred: np.ndarray, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    p_pred = np.asarray(p_pred, dtype=float).reshape(-1)
    y_hat = (p_pred >= threshold).astype(float)
    return float(np.mean(y_hat == y_true))


def log_loss_manual(y_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    p_pred = np.asarray(p_pred, dtype=float).reshape(-1)
    p_pred = np.clip(p_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p_pred) + (1.0 - y_true) * np.log(1.0 - p_pred)))


def loglik_manual(y_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    p_pred = np.asarray(p_pred, dtype=float).reshape(-1)
    p_pred = np.clip(p_pred, eps, 1.0 - eps)
    return float(np.sum(y_true * np.log(p_pred) + (1.0 - y_true) * np.log(1.0 - p_pred)))


def confusion_metrics(y_true: np.ndarray, p_pred: np.ndarray, threshold: float = 0.5):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_hat = (np.asarray(p_pred).reshape(-1) >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_hat == 1)))
    tn = int(np.sum((y_true == 0) & (y_hat == 0)))
    fp = int(np.sum((y_true == 0) & (y_hat == 1)))
    fn = int(np.sum((y_true == 1) & (y_hat == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
    }


def roc_curve_manual(y_true: np.ndarray, p_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    p_pred = np.asarray(p_pred, dtype=float).reshape(-1)

    thresholds = np.unique(p_pred)[::-1]
    thresholds = np.r_[1.0 + 1e-12, thresholds, -1e-12]

    rows = []
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    for thr in thresholds:
        y_hat = (p_pred >= thr).astype(int)
        tp = np.sum((y_true == 1) & (y_hat == 1))
        fp = np.sum((y_true == 0) & (y_hat == 1))

        tpr = tp / n_pos if n_pos > 0 else np.nan
        fpr = fp / n_neg if n_neg > 0 else np.nan

        rows.append({"threshold": thr, "tpr": tpr, "fpr": fpr})

    roc = pd.DataFrame(rows).drop_duplicates(subset=["fpr", "tpr"]).sort_values(["fpr", "tpr"])
    return roc


def auc_manual(y_true: np.ndarray, p_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    p_pred = np.asarray(p_pred, dtype=float).reshape(-1)

    if np.unique(y_true).size < 2:
        return np.nan

    order = np.argsort(p_pred)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p_pred) + 1)

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    sum_ranks_pos = ranks[y_true == 1].sum()

    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def reliability_curve(y_true, p_pred, n_bins=10, strategy="quantile"):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    p_pred = np.asarray(p_pred, dtype=float).reshape(-1)

    df = pd.DataFrame({"y": y_true, "p": p_pred})

    if strategy == "quantile":
        df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    elif strategy == "uniform":
        bins = np.linspace(0, 1, n_bins + 1)
        df["bin"] = pd.cut(df["p"], bins=bins, include_lowest=True)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    rel = (
        df.groupby("bin", observed=False)
        .agg(
            count=("y", "size"),
            p_mean=("p", "mean"),
            obs_freq=("y", "mean")
        )
        .reset_index()
    )
    return rel


def expected_calibration_error(y_true, p_pred, n_bins=10, strategy="quantile"):
    rel = reliability_curve(y_true, p_pred, n_bins=n_bins, strategy=strategy)
    if rel.shape[0] == 0:
        return np.nan
    n = rel["count"].sum()
    if n == 0:
        return np.nan
    return float(np.sum((rel["count"] / n) * np.abs(rel["obs_freq"] - rel["p_mean"])))


# This function gathers the main validation metrics for one set of predictions
# It also includes quantities used in logistic model comparison
def evaluate_occurrence_predictions(
    y_true,
    p_pred,
    p_null=None,
    n_params=None,
    threshold=0.5,
    calib_bins=10,
):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    p_pred = np.asarray(p_pred, dtype=float).reshape(-1)

    ll_model = loglik_manual(y_true, p_pred)

    out = {
        "n": len(y_true),
        "rain_freq": float(y_true.mean()),
        "mean_pred": float(p_pred.mean()),
        "brier": brier_score_manual(y_true, p_pred),
        "logloss": log_loss_manual(y_true, p_pred),
        "accuracy_050": accuracy_manual(y_true, p_pred, threshold=threshold),
        "auc": auc_manual(y_true, p_pred),
        "ece_quantile": expected_calibration_error(y_true, p_pred, n_bins=calib_bins, strategy="quantile"),
        "ece_uniform": expected_calibration_error(y_true, p_pred, n_bins=calib_bins, strategy="uniform"),
        "loglik_model": ll_model,
    }

    out.update(confusion_metrics(y_true, p_pred, threshold=threshold))

    if p_null is not None:
        p_null = np.asarray(p_null, dtype=float).reshape(-1)
        ll_null = loglik_manual(y_true, p_null)
        out["loglik_null"] = ll_null
        out["lr_chi2"] = float(2.0 * (ll_model - ll_null))
        out["mcfadden_r2"] = float(1.0 - ll_model / ll_null) if ll_null != 0 else np.nan

    if n_params is not None:
        out["aic"] = float(-2.0 * ll_model + 2.0 * n_params)
        out["bic"] = float(-2.0 * ll_model + n_params * np.log(len(y_true)))

    return out


# Better validation figure with three panels:
# calibration, ROC, and score distribution
def plot_validation_summary(y_true, p_pred, title_prefix="", filename=None):
    y_true = np.asarray(y_true).reshape(-1)
    p_pred = np.asarray(p_pred).reshape(-1)

    rel = reliability_curve(y_true, p_pred, n_bins=10, strategy="quantile")
    roc = roc_curve_manual(y_true, p_pred)
    auc = auc_manual(y_true, p_pred)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # calibration panel
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    ax.plot(rel["p_mean"], rel["obs_freq"], marker="o")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed rain frequency")
    ax.set_title(f"{title_prefix} calibration")
    ax.grid(True, alpha=0.3)

    # ROC panel
    ax = axes[1]
    ax.plot(roc["fpr"], roc["tpr"], label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{title_prefix} ROC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # distribution panel
    ax = axes[2]
    ax.hist(p_pred[y_true == 0], bins=40, alpha=0.6, density=True, label="no rain")
    ax.hist(p_pred[y_true == 1], bins=40, alpha=0.6, density=True, label="rain")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title(f"{title_prefix} score distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

    return rel, roc