#%%
# Imports and paths
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

DATA_FOLDER = os.environ.get("DATA_FOLDER", "../phd_extremes/data/")
IM_FOLDER = os.path.join(DATA_FOLDER, "../phd_extremes/thesis/resources/images/downscaling/")
DOWNSCALING_TABLE = os.path.join(DATA_FOLDER, "downscaling/downscaling_table_named_2019_2024.csv")

os.makedirs(IM_FOLDER, exist_ok=True)

#%%
# Predictor groups used in the occurrence model
TIME_COLS = ["tod_sin", "tod_cos", "doy_sin", "doy_cos", "month_sin", "month_cos"]
SPATIAL_COLS = ["lat_Y", "lon_Y", "lat_X", "lon_X"]

#%%
# Read the 27 pixel predictors and keep a stable order
def get_x_cols27_downscaling(df: pd.DataFrame) -> list[str]:
    pat = re.compile(r"^X_p(\d{2})_dt(-1h|0h|\+1h)$")
    cols = [c for c in df.columns if pat.match(c)]
    order_dt = {"-1h": 0, "0h": 1, "+1h": 2}

    def key(c):
        m = pat.match(c)
        return (int(m.group(1)), order_dt[m.group(2)])

    return sorted(cols, key=key)

#%%
# Build one blocked train/validation split
# This avoids mixing nearby times between train and validation
def make_split_blocked(df: pd.DataFrame, train_frac=0.8, seed=1, block="30D"):
    rng = np.random.default_rng(seed)

    d = df.copy()
    d["block"] = d["time"].dt.floor(block)

    blocks = np.array(sorted(d["block"].unique()))
    rng.shuffle(blocks)

    n_train = int(np.floor(train_frac * len(blocks)))
    train_blocks = set(blocks[:n_train])

    train_idx = d.index[d["block"].isin(train_blocks)].to_numpy()
    valid_idx = d.index[~d["block"].isin(train_blocks)].to_numpy()

    return {
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "train_blocks": train_blocks,
    }

#%%
# Build blocked cross-validation splits
def make_blocked_cv_splits(df: pd.DataFrame, n_splits=5, block="30D", seed=1):
    rng = np.random.default_rng(seed)

    d = df.copy()
    d["block"] = d["time"].dt.floor(block)

    unique_blocks = np.array(sorted(d["block"].unique()))
    rng.shuffle(unique_blocks)

    folds = np.array_split(unique_blocks, n_splits)

    splits = []
    for k in range(n_splits):
        valid_blocks = set(folds[k])
        train_idx = d.index[~d["block"].isin(valid_blocks)].to_numpy()
        valid_idx = d.index[d["block"].isin(valid_blocks)].to_numpy()

        splits.append({
            "fold": k,
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "valid_blocks": sorted(valid_blocks),
        })

    return splits

#%%
# Standardize predictors using training data only
# This prevents leakage from validation into train preprocessing
def standardize_train_only(df: pd.DataFrame, train_idx_labels: np.ndarray, scale_cols: list[str]):
    df2 = df.copy()
    df2[scale_cols] = df2[scale_cols].astype(float)

    mu = df2.loc[train_idx_labels, scale_cols].mean(axis=0, skipna=True)
    sdv = df2.loc[train_idx_labels, scale_cols].std(axis=0, skipna=True, ddof=1).replace(0.0, 1.0)

    df2[scale_cols] = (df2[scale_cols] - mu) / sdv
    return df2, mu, sdv

#%%
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

#%%
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

#%%
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

#%%
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

#%%
# Build the occurrence dataset from raw data
# This is where we define the binary event of interest:
# rain occurrence = 1 if Y_obs > 0, else 0
def prepare_occurrence_dataframe(
    df_raw: pd.DataFrame,
    use_time: bool = True,
    use_spatial: bool = True,
    use_summaries: bool = True,
    use_cube: bool = True,
):
    df = df_raw.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)

    x_cols27 = get_x_cols27_downscaling(df)

    # keep rows with observed target
    df = df.loc[df["Y_obs"].notna()].copy()

    # remove obvious inconsistencies
    X_block = df[x_cols27].to_numpy(dtype=float)
    x_cube_sum = X_block.sum(axis=1)
    incoherent = (df["Y_obs"] > 0) & (x_cube_sum == 0)
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
    df["radar_max"] = X_block.max(axis=1)
    df["radar_mean"] = X_block.mean(axis=1)
    df["radar_sum"] = X_block.sum(axis=1)

    x_cols = []
    if use_time:
        x_cols += TIME_COLS
    if use_spatial:
        x_cols += SPATIAL_COLS
    if use_summaries:
        x_cols += ["radar_max", "radar_mean", "radar_sum"]
    if use_cube:
        x_cols += x_cols27

    keep_cols = ["time", "station", "Y_obs"] + x_cols
    df = df[keep_cols].copy()

    return df, x_cols27, x_cols

#%%
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

#%%
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

#%%
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
    plt.figure(figsize=(6, 4))
    plt.plot(fit["train_loss"], label="train")
    if len(fit["valid_loss"]) > 0:
        plt.plot(fit["valid_loss"], label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

#%%
# Load raw data
df_raw = pd.read_csv(DOWNSCALING_TABLE, sep=";")
df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True)

print("Missing values in Y_obs:", df_raw["Y_obs"].isna().sum())
print("Proportion of Y_obs = 0:", (df_raw["Y_obs"] == 0).mean())

#%%
# Build the binary occurrence dataset
df_occ, x_cols27, x_cols = prepare_occurrence_dataframe(
    df_raw,
    use_time=True,
    use_spatial=True,
    use_summaries=True,
    use_cube=True,
)

print("Occurrence dataset shape:", df_occ.shape)
print("Number of predictors:", len(x_cols))

#%%
# Create a blocked train/validation split
split = make_split_blocked(df_occ, train_frac=0.8, seed=2, block="30D")
print("n_train:", len(split["train_idx"]), "n_valid:", len(split["valid_idx"]))

#%%
# Standardize predictors using train only
df_std, mu, sdv = standardize_train_only(df_occ, split["train_idx"], x_cols)

built = build_Xy_occurrence(
    df_std=df_std,
    x_cols=x_cols,
    train_idx_labels=split["train_idx"],
    valid_idx_labels=split["valid_idx"],
)

print("X_train shape:", built["X_train"].shape)
print("Rain frequency train:", built["y_train"].mean())
print("Rain frequency valid:", built["y_valid"].mean())

#%%
# Fit the logistic occurrence model
fit = train_logit_model(
    X_train=built["X_train"],
    y_train=built["y_train"],
    X_valid=built["X_valid"],
    y_valid=built["y_valid"],
    lr=1e-2,
    n_epochs=400,
    seed=1,
    device=None,
    patience=40,
)

print("Stopped epoch:", fit["stopped_epoch"])

#%%
# Inspect fitted coefficients and odds ratios
with torch.no_grad():
    bias_hat = fit["model"].linear.bias.detach().cpu().item()
    w = fit["model"].linear.weight.detach().cpu().numpy().ravel()

coef_df = pd.DataFrame({
    "feature": x_cols,
    "coef_logit": w,
    "odds_ratio_1sd": np.exp(w),
    "abs_coef": np.abs(w)
}).sort_values("abs_coef", ascending=False)

print("Estimated intercept:", bias_hat)
print("Intercept probability:", 1 / (1 + np.exp(-bias_hat)))
print(coef_df.head(20))

#%%
plot_loss_history(
    fit,
    title="Occurrence logistic model loss",
    filename=os.path.join(IM_FOLDER, "occurrence_logit_loss.png")
)

#%%
# Predict on validation
y_valid = built["y_valid"].reshape(-1)
p_valid = predict_occurrence_probability(fit["model"], built["X_valid"])

# Build the intercept-only baseline on the same validation set
p0 = float(built["y_train"].mean())
p_valid_base = np.full_like(y_valid, fill_value=p0, dtype=float)

# Number of free parameters in the fitted logistic model
# one intercept + one coefficient per predictor
n_params_model = 1 + built["X_train"].shape[1]

metrics_valid = evaluate_occurrence_predictions(
    y_true=y_valid,
    p_pred=p_valid,
    p_null=p_valid_base,
    n_params=n_params_model,
    threshold=0.5,
    calib_bins=10
)

metrics_base = evaluate_occurrence_predictions(
    y_true=y_valid,
    p_pred=p_valid_base,
    p_null=p_valid_base,
    n_params=1,
    threshold=0.5,
    calib_bins=10
)

print("Validation metrics for the fitted model")
for k, v in metrics_valid.items():
    print(f"{k}: {v}")

print("\nValidation metrics for the constant baseline")
for k, v in metrics_base.items():
    print(f"{k}: {v}")

#%%
# Better validation graph
rel_valid, roc_valid = plot_validation_summary(
    y_true=y_valid,
    p_pred=p_valid,
    title_prefix="Validation",
    filename=os.path.join(IM_FOLDER, "occurrence_validation_summary.png")
)

#%%
# Add probabilities back to the full dataset
p_all = predict_occurrence_probability(fit["model"], built["X_all"])

df_occ_pred = df_occ.copy()
df_occ_pred["p_occ_hat"] = p_all
df_occ_pred["is_rain"] = (df_occ_pred["Y_obs"] > 0).astype(int)

print(df_occ_pred[["time", "station", "Y_obs", "is_rain", "p_occ_hat"]].head())

#%%
# Blocked cross-validation
K = 5
cv_splits = make_blocked_cv_splits(df_occ, n_splits=K, block="30D", seed=1)

rows_cv = []

for sp in cv_splits:
    df_std_cv, mu_cv, sdv_cv = standardize_train_only(df_occ, sp["train_idx"], x_cols)

    built_cv = build_Xy_occurrence(
        df_std=df_std_cv,
        x_cols=x_cols,
        train_idx_labels=sp["train_idx"],
        valid_idx_labels=sp["valid_idx"],
    )

    fit_cv = train_logit_model(
        X_train=built_cv["X_train"],
        y_train=built_cv["y_train"],
        X_valid=built_cv["X_valid"],
        y_valid=built_cv["y_valid"],
        lr=1e-2,
        n_epochs=300,
        seed=1,
        device=None,
        patience=30,
    )

    y_valid_cv = built_cv["y_valid"].reshape(-1)
    p_valid_cv = predict_occurrence_probability(fit_cv["model"], built_cv["X_valid"])

    p0_cv = float(built_cv["y_train"].mean())
    p_valid_base_cv = np.full_like(y_valid_cv, fill_value=p0_cv, dtype=float)

    n_params_cv = 1 + built_cv["X_train"].shape[1]

    metrics_cv = evaluate_occurrence_predictions(
        y_true=y_valid_cv,
        p_pred=p_valid_cv,
        p_null=p_valid_base_cv,
        n_params=n_params_cv,
        threshold=0.5,
        calib_bins=10
    )

    metrics_base_cv = evaluate_occurrence_predictions(
        y_true=y_valid_cv,
        p_pred=p_valid_base_cv,
        p_null=p_valid_base_cv,
        n_params=1,
        threshold=0.5,
        calib_bins=10
    )

    rows_cv.append({
        "fold": sp["fold"],
        "n_valid": metrics_cv["n"],
        "rain_freq_valid": metrics_cv["rain_freq"],
        "mean_pred_valid": metrics_cv["mean_pred"],
        "brier_model": metrics_cv["brier"],
        "brier_baseline": metrics_base_cv["brier"],
        "logloss_model": metrics_cv["logloss"],
        "logloss_baseline": metrics_base_cv["logloss"],
        "auc_model": metrics_cv["auc"],
        "auc_baseline": metrics_base_cv["auc"],
        "acc_model": metrics_cv["accuracy_050"],
        "acc_baseline": metrics_base_cv["accuracy_050"],
        "ece_model": metrics_cv["ece_quantile"],
        "ece_baseline": metrics_base_cv["ece_quantile"],
        "mcfadden_r2": metrics_cv["mcfadden_r2"],
        "lr_chi2": metrics_cv["lr_chi2"],
        "aic_model": metrics_cv["aic"],
        "bic_model": metrics_cv["bic"],
        "stopped_epoch": fit_cv["stopped_epoch"],
    })

df_cv = pd.DataFrame(rows_cv)
print(df_cv)

#%%
# Summarize CV performance
summary_cv = pd.DataFrame({
    "metric": [
        "brier_model",
        "brier_baseline",
        "logloss_model",
        "logloss_baseline",
        "auc_model",
        "acc_model",
        "ece_model",
        "mcfadden_r2",
        "lr_chi2",
        "aic_model",
        "bic_model",
    ],
    "mean": [
        df_cv["brier_model"].mean(),
        df_cv["brier_baseline"].mean(),
        df_cv["logloss_model"].mean(),
        df_cv["logloss_baseline"].mean(),
        df_cv["auc_model"].mean(),
        df_cv["acc_model"].mean(),
        df_cv["ece_model"].mean(),
        df_cv["mcfadden_r2"].mean(),
        df_cv["lr_chi2"].mean(),
        df_cv["aic_model"].mean(),
        df_cv["bic_model"].mean(),
    ],
    "std": [
        df_cv["brier_model"].std(ddof=1),
        df_cv["brier_baseline"].std(ddof=1),
        df_cv["logloss_model"].std(ddof=1),
        df_cv["logloss_baseline"].std(ddof=1),
        df_cv["auc_model"].std(ddof=1),
        df_cv["acc_model"].std(ddof=1),
        df_cv["ece_model"].std(ddof=1),
        df_cv["mcfadden_r2"].std(ddof=1),
        df_cv["lr_chi2"].std(ddof=1),
        df_cv["aic_model"].std(ddof=1),
        df_cv["bic_model"].std(ddof=1),
    ],
})
print(summary_cv)

#%%
# Check how many incoherent rows were removed
df_tmp = df_raw.loc[df_raw["Y_obs"].notna()].copy()

x_cols27_tmp = get_x_cols27_downscaling(df_tmp)
X_block_tmp = df_tmp[x_cols27_tmp].to_numpy(dtype=float)
x_cube_sum_tmp = X_block_tmp.sum(axis=1)

mask_incoherent = (df_tmp["Y_obs"] > 0) & (x_cube_sum_tmp == 0)

n_total = len(df_tmp)
n_removed = int(mask_incoherent.sum())
prop_removed = n_removed / n_total

print("Total observations:", n_total)
print("Removed observations:", n_removed)
print("Proportion removed:", prop_removed)

mask_positive = df_tmp["Y_obs"] > 0
n_positive = int(mask_positive.sum())
n_removed_among_positive = int(mask_incoherent.sum())
prop_removed_among_positive = n_removed_among_positive / n_positive if n_positive > 0 else np.nan

print("Positive-rain observations:", n_positive)
print("Removed among positive-rain cases:", n_removed_among_positive)
print("Proportion removed among positive-rain cases:", prop_removed_among_positive)

removed_by_station = (
    df_tmp.loc[mask_incoherent]
    .groupby("station")
    .size()
    .sort_values(ascending=False)
)

print(removed_by_station)