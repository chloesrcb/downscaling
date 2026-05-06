from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from downscaling.config import TIME_COLS, SPATIAL_COLS
from downscaling.data import get_x_cols27_downscaling

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