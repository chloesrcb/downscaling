from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from downscaling.plotting import PLOT_DPI, clean_figure, configure_plot_style, save_png
from downscaling.settings import TIME_COLS, SPATIAL_COLS
from downscaling.data import get_x_cols27_downscaling


configure_plot_style()

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
    ax.set_xlabel(r"Mean predicted $P(X_{\mathbf{s},t} > 0 \mid \mathbf{C}_{\mathbf{s},t})$")
    ax.set_ylabel(r"Observed frequency of $X_{\mathbf{s},t} > 0$")
    ax.grid(True, alpha=0.3)

    # ROC panel
    ax = axes[1]
    ax.plot(roc["fpr"], roc["tpr"], label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # distribution panel
    ax = axes[2]
    ax.hist(p_pred[y_true == 0], bins=40, alpha=0.6, density=True, label="no rain")
    ax.hist(p_pred[y_true == 1], bins=40, alpha=0.6, density=True, label="rain")
    ax.set_xlabel(r"Predicted $P(X_{\mathbf{s},t} > 0 \mid \mathbf{C}_{\mathbf{s},t})$")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    clean_figure(fig)
    plt.tight_layout()

    if filename is not None:
        save_png(fig, filename, dpi=PLOT_DPI)
    plt.show()

    return rel, roc

def plot_roc_curve(y_true, p_pred, title="", filename=None):
    roc = roc_curve_manual(y_true, p_pred)
    auc = auc_manual(y_true, p_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(roc["fpr"], roc["tpr"], label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    # plt.title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if filename is not None:
        save_png(fig, filename, dpi=PLOT_DPI)
    plt.show()


# Build X and y for occurrence
# y = 1 if rain is observed, 0 otherwise
def build_Xy_occurrence(
    df_std: pd.DataFrame,
    x_cols: list[str],
    train_idx_labels: np.ndarray,
    valid_idx_labels: np.ndarray,
):
    X = df_std[x_cols].to_numpy(dtype=np.float32)
    y = (df_std["Y_obs"].to_numpy(dtype=np.float32) > 0).astype(np.float32)

    tr_pos = df_std.index.get_indexer(train_idx_labels)
    va_pos = df_std.index.get_indexer(valid_idx_labels)

    if (tr_pos < 0).any() or (va_pos < 0).any():
        raise ValueError("Some split indices were not found in df_std.index.")

    return {
        "X_all": X,
        "y_all": y,
        "tr_pos": tr_pos,
        "va_pos": va_pos,
        "X_train": X[tr_pos],
        "y_train": y[tr_pos].reshape(-1, 1),
        "X_valid": X[va_pos],
        "y_valid": y[va_pos].reshape(-1, 1),
    }


# Build the occurrence dataset from raw data
# This is where we define the binary event of interest:
# rain occurrence = 1 if Y_obs > 0, else 0
def prepare_occurrence_dataframe(
    df_raw: pd.DataFrame,
    use_time: bool = True,
    use_spatial: bool = True,
    use_summaries: bool = True,
    use_cube: bool = True,
    remove_incoherent: bool = True,
    summary_scale: str = "log", # "raw" or "log" or "both"
):
    df = df_raw.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)

    x_cols27 = get_x_cols27_downscaling(df)

    # keep rows with observed target
    df = df.loc[df["Y_obs"].notna()].copy()

    X_block = df[x_cols27].to_numpy(dtype=float)
    x_cube_sum = X_block.sum(axis=1)

    incoherent = (df["Y_obs"] > 0) & (x_cube_sum == 0)

    if remove_incoherent:
        df = df.loc[~incoherent].copy()

    # rebuild radar arrays after filtering
    x_cols27 = get_x_cols27_downscaling(df)
    X_block = df[x_cols27].to_numpy(dtype=float)

    # cyclic time features
    df["hour"] = df["time"].dt.hour
    df["minute"] = df["time"].dt.minute
    df["month"] = df["time"].dt.month

    tod = df["hour"] * 60 + df["minute"]
    doy = df["time"].dt.dayofyear

    df["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    df["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    df["doy_sin"] = np.sin(2 * np.pi * (doy - 1) / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * (doy - 1) / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    # simple summaries of the radar cube
   # summaries of raw radar cube
    df["radar_max"] = X_block.max(axis=1)
    df["radar_mean"] = X_block.mean(axis=1)
    df["radar_sum"] = X_block.sum(axis=1)

    # log-transform summaries
    df["log_radar_max"] = np.log1p(df["radar_max"])
    df["log_radar_mean"] = np.log1p(df["radar_mean"])
    df["log_radar_sum"] = np.log1p(df["radar_sum"])

    # detection indicators
    df["radar_any"] = (df["radar_sum"] > 0).astype(int)
    df["radar_center"] = (df[x_cols27[2]] > 0).astype(int)

    x_cols = []
    if use_time:
        x_cols += TIME_COLS
    if use_spatial:
        x_cols += SPATIAL_COLS
    if use_summaries:
        if summary_scale not in ["raw", "log", "both"]:
            raise ValueError("summary_scale must be 'raw', 'log', or 'both'.")

        if summary_scale in ["raw", "both"]:
            x_cols += [
                "radar_max",
                "radar_mean",
                "radar_sum",
            ]

        if summary_scale in ["log", "both"]:
            x_cols += [
                "log_radar_max",
                "log_radar_mean",
                "log_radar_sum",
            ]

        x_cols += [
            "radar_any",
            "radar_center",
        ]
    if use_cube:
        x_cols += x_cols27

    keep_cols = ["time", "station", "Y_obs"] + x_cols
    df = df[keep_cols].copy()

    return df, x_cols27, x_cols


# This is a plain logistic regression written in PyTorch
# The output is one logit, transformed later into a probability
class OccurrenceLogit(nn.Module):
    def __init__(self, d_in: int, init_logit: Optional[float] = None):
        super().__init__()
        self.linear = nn.Linear(d_in, 1)

        # start from an intercept-only model
        nn.init.zeros_(self.linear.weight)

        if init_logit is None:
            nn.init.zeros_(self.linear.bias)
        else:
            with torch.no_grad():
                self.linear.bias.fill_(float(init_logit))

    def forward(self, x):
        return self.linear(x)
    


# Fit the logistic model with early stopping on validation loss
def train_logit_model(
    X_train,
    y_train,
    X_valid=None,
    y_valid=None,
    lr=1e-2,
    n_epochs=400,
    seed=1,
    device=None,
    patience=40,
    min_delta=1e-5,
):
    torch.manual_seed(seed)

    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    X_train = torch.as_tensor(X_train, dtype=torch.float32, device=dev)
    y_train = torch.as_tensor(y_train, dtype=torch.float32, device=dev).reshape(-1, 1)

    if X_valid is not None:
        X_valid = torch.as_tensor(X_valid, dtype=torch.float32, device=dev)
        y_valid = torch.as_tensor(y_valid, dtype=torch.float32, device=dev).reshape(-1, 1)

    # initialize the intercept at the training rain frequency
    p0 = float(y_train.mean().item())
    p0 = np.clip(p0, 1e-6, 1.0 - 1e-6)
    init_logit = np.log(p0 / (1.0 - p0))

    print(f"Train rain frequency = {p0:.6f}")
    print(f"Initial intercept logit = {init_logit:.4f}")

    model = OccurrenceLogit(X_train.shape[1], init_logit=init_logit).to(dev)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_hist = []
    valid_loss_hist = []

    best_state = None
    best_score = float("inf")
    bad_epochs = 0
    stopped_epoch = n_epochs

    for epoch in range(n_epochs):
        model.train()

        logits = model(X_train)
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = float(loss.detach().cpu())
        train_loss_hist.append(train_loss)

        if X_valid is not None:
            model.eval()
            with torch.no_grad():
                logits_valid = model(X_valid)
                valid_loss = float(criterion(logits_valid, y_valid).detach().cpu())
            valid_loss_hist.append(valid_loss)
            score = valid_loss
        else:
            score = train_loss

        improved = (best_score - score) > min_delta
        if improved:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if (epoch == 0) or ((epoch + 1) % 50 == 0):
            if X_valid is None:
                print(f"epoch {epoch+1:4d} | train loss = {train_loss:.6f}")
            else:
                print(f"epoch {epoch+1:4d} | train loss = {train_loss:.6f} | valid loss = {valid_loss:.6f}")

        if (X_valid is not None) and (bad_epochs >= patience):
            stopped_epoch = epoch + 1
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "train_loss": train_loss_hist,
        "valid_loss": valid_loss_hist,
        "init_logit": float(init_logit),
        "train_freq": float(p0),
        "best_score": float(best_score),
        "stopped_epoch": int(stopped_epoch),
    }


# Turn logits into probabilities
def predict_occurrence_probability(model, X):
    dev = next(model.parameters()).device
    X = torch.as_tensor(X, dtype=torch.float32, device=dev)

    model.eval()
    with torch.no_grad():
        logits = model(X).reshape(-1).detach().cpu().numpy()

    return sigmoid_np(logits)


def predict_occurrence_logit(model, X):
    dev = next(model.parameters()).device
    X = torch.as_tensor(X, dtype=torch.float32, device=dev)

    model.eval()
    with torch.no_grad():
        logits = model(X).reshape(-1).detach().cpu().numpy()

    return logits


def plot_loss_history(fit: dict, title: str, filename: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fit["train_loss"], label="train")
    if len(fit["valid_loss"]) > 0:
        ax.plot(fit["valid_loss"], label="valid")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    clean_figure(fig)
    plt.tight_layout()

    if filename is not None:
        save_png(fig, filename, dpi=PLOT_DPI)
    plt.show()
