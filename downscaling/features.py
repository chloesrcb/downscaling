
import numpy as np
import pandas as pd


def standardize_train_only(
    df: pd.DataFrame,
    train_idx_labels: np.ndarray,
    scale_cols: list[str],
):
    """
    Standardize covariates using training observations only.
    """
    df2 = df.copy()
    df2[scale_cols] = df2[scale_cols].astype(float)

    mu = df2.loc[train_idx_labels, scale_cols].mean(axis=0, skipna=True)

    sdv = (
        df2.loc[train_idx_labels, scale_cols]
        .std(axis=0, skipna=True, ddof=1)
        .replace(0.0, 1.0)
    )

    df2[scale_cols] = (df2[scale_cols] - mu) / sdv

    return df2, mu, sdv


def build_xy_train_valid(
    df_std: pd.DataFrame,
    x_cols: list[str],
    train_idx_labels,
    valid_idx_labels,
):
    X = df_std[x_cols].to_numpy(dtype=np.float32)
    Y = df_std["Y_obs"].to_numpy(dtype=np.float32)

    X_all = X.reshape(len(X), 1, X.shape[1])

    tr_pos = df_std.index.get_indexer(train_idx_labels)
    va_pos = df_std.index.get_indexer(valid_idx_labels)

    if (tr_pos < 0).any() or (va_pos < 0).any():
        raise ValueError("Split indices not found in df_std.index.")

    return {
        "X_all": X_all,
        "Y_all": Y,
        "tr_pos": tr_pos,
        "va_pos": va_pos,
        "X_train": X_all[tr_pos],
        "Y_train": Y[tr_pos].reshape(-1, 1),
        "X_valid": X_all[va_pos],
        "Y_valid": Y[va_pos].reshape(-1, 1),
    }


def build_X_from_meta(df: pd.DataFrame, meta: dict) -> np.ndarray:
    df2 = df.copy()
    x_cols = meta["x_cols"]

    df2[x_cols] = df2[x_cols].astype(float)
    df2[x_cols] = (
        df2[x_cols]
        - meta["mu"][x_cols]
    ) / meta["sdv"][x_cols].replace(0.0, 1.0)

    X = df2[x_cols].to_numpy(np.float32)

    return X.reshape(len(df2), 1, -1)
    