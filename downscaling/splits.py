import numpy as np
import pandas as pd


def make_time_blocks(df: pd.DataFrame, block: str = "30D") -> pd.Series:
    """
    Create time blocks for blocked cross-validation.
    They are randomly selected.
    """
    return df["time"].dt.floor(block)


def make_blocked_cv_splits(df: pd.DataFrame, n_splits: int = 5,
                            block: str = "30D",
                            seed: int = 1,):
    """
    Create blocked cross-validation splits based on time blocks.
    Blocks are randomly selected and assigned to folds.
    """
    rng = np.random.default_rng(seed)

    block_labels = make_time_blocks(df, block=block)
    unique_blocks = np.array(sorted(block_labels.unique()))
    rng.shuffle(unique_blocks)

    folds = np.array_split(unique_blocks, n_splits)

    splits = []
    for k in range(n_splits):
        valid_blocks = set(folds[k])
        train_idx = df.index[~block_labels.isin(valid_blocks)].to_numpy()
        valid_idx = df.index[block_labels.isin(valid_blocks)].to_numpy()

        splits.append({
            "fold": k,
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "valid_blocks": sorted(valid_blocks),
        })

    return splits


def make_single_split_from_train(
                                df_train: pd.DataFrame,
                                train_frac: float = 0.8,
                                block: str = "30D",
                                seed: int = 123,
                            ):
    """
    Make a single train/validation split from the training data, using time blocks.
    """
    d = df_train.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True)

    block_labels = d["time"].dt.floor(block)
    unique_blocks = np.array(sorted(block_labels.unique()))

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_blocks)

    n_train_blocks = int(np.floor(train_frac * len(unique_blocks)))
    n_train_blocks = max(1, min(n_train_blocks, len(unique_blocks) - 1))

    inner_train_blocks = set(unique_blocks[:n_train_blocks])

    inner_train_idx = d.index[block_labels.isin(inner_train_blocks)].to_numpy()
    inner_valid_idx = d.index[~block_labels.isin(inner_train_blocks)].to_numpy()

    return {
        "fold": 0,
        "train_idx": inner_train_idx,
        "valid_idx": inner_valid_idx,
        "valid_blocks": sorted(set(unique_blocks[n_train_blocks:])),
    }



def make_train_valid_test_split(
    df: pd.DataFrame,
    test_frac: float = 0.10,
    block: str = "30D",
    seed: int = 2026,
):
    """
    Split df into train_valid and test using time blocks.

    The test set is removed before any CV/tuning/model selection.
    The fraction is block-based, therefore the observation fraction is approximate.
    """
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True)

    block_labels = make_time_blocks(d, block=block)
    unique_blocks = np.array(sorted(block_labels.unique()))

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_blocks)

    n_test_blocks = int(np.ceil(test_frac * len(unique_blocks)))
    n_test_blocks = max(1, min(n_test_blocks, len(unique_blocks) - 1))

    test_blocks = set(unique_blocks[:n_test_blocks])
    train_valid_blocks = set(unique_blocks[n_test_blocks:])

    train_valid_idx = d.index[block_labels.isin(train_valid_blocks)].to_numpy()
    test_idx = d.index[block_labels.isin(test_blocks)].to_numpy()

    df_train_valid = d.loc[train_valid_idx].copy()
    df_test = d.loc[test_idx].copy()

    info = {
        "test_frac_requested": test_frac,
        "test_frac_observed": len(df_test) / len(d),
        "n_total": len(d),
        "n_train_valid": len(df_train_valid),
        "n_test": len(df_test),
        "n_blocks_total": len(unique_blocks),
        "n_blocks_train_valid": len(train_valid_blocks),
        "n_blocks_test": len(test_blocks),
        "test_blocks": sorted(test_blocks),
        "train_valid_blocks": sorted(train_valid_blocks),
    }

    return df_train_valid, df_test, info

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