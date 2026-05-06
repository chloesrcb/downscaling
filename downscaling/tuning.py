from itertools import product
from typing import Any, Dict, Optional

import pandas as pd

from downscaling.config import SIGMA_INIT, KAPPA_INIT
from downscaling.splits import make_single_split_from_train
from downscaling.nn import Config, parse_widths, predict_params_on_df_variant
from downscaling.evaluation import (
    evaluate_nn_config_on_split,
    evaluate_nn_config_on_split_fast,
)
from downscaling.regression import get_gam_initial_values
from downscaling.scores import summarize_distribution_metrics, smad_exponential_margins
from downscaling.egpd import egpd_left_censored_nll_sum



def tune_nn_on_outer_train(
    df_model: pd.DataFrame,
    outer_split: Dict[str, Any],
    x_sets: dict[str, list[str]],
    param_grid: dict,
    seed: int = 1,
    device: Optional[str] = None,
):
    df_outer_train = df_model.loc[outer_split["train_idx"]].copy()

    inner_split = make_single_split_from_train(
        df_outer_train,
        train_frac=0.8,
        block="30D",
        seed=seed + 100,
    )

    rows = []

    common_keys = [
        "variant",
        "x_set_name",
        "widths",
        "lr",
        "weight_decay",
        "batch_size",
        "n_ep",
        "xi_init",
        "censor_threshold",
        "init_source",
    ]

    common_values = [param_grid[k] for k in common_keys]

    for vals in product(*common_values):
        params = dict(zip(common_keys, vals))

        variant = params["variant"]
        x_set_name = params["x_set_name"]
        init_source = params["init_source"]

        if x_set_name not in x_sets:
            raise ValueError(f"Unknown x_set_name: {x_set_name}")

        x_cols = x_sets[x_set_name]

        if variant == "both":
            sigma_candidates = param_grid["sigma_init"]
            kappa_candidates = param_grid["kappa_init"]

        elif variant == "sigma_only":
            sigma_candidates = param_grid["sigma_init"]
            kappa_candidates = [KAPPA_INIT]

        elif variant == "kappa_only":
            sigma_candidates = [SIGMA_INIT]
            kappa_candidates = param_grid["kappa_init"]

        else:
            raise ValueError(f"Unknown variant: {variant}")

        gam_init = None

        if init_source == "gam":
            try:
                gam_init = get_gam_initial_values(
                    df_train=df_outer_train.loc[inner_split["train_idx"]],
                    covariate_col=single_cov_col,
                    variant=variant,
                    xi_fixed=float(params["xi_init"]),
                    sigma_init=SIGMA_INIT,
                    kappa_init=KAPPA_INIT,
                    censor_threshold=float(params["censor_threshold"]),
                )

                sigma_candidates = [gam_init["sigma_init_gam"]]
                kappa_candidates = [gam_init["kappa_init_gam"]]

            except Exception as e:
                print(f"GAM init failed for {params}: {e}")
                continue

        for sigma_init, kappa_init in product(sigma_candidates, kappa_candidates):
            params_variant = params.copy()

            params_variant["sigma_init"] = float(sigma_init)
            params_variant["kappa_init"] = float(kappa_init)
            params_variant["n_covariates"] = len(x_cols)

            if gam_init is not None:
                params_variant.update(gam_init)

            cfg = Config(
                name=f"nn_{x_set_name}_{variant}_{params_variant['widths']}_{init_source}",
                widths=parse_widths(params_variant["widths"]),
            )

            res = evaluate_nn_config_on_split_fast(
                df_model=df_model,
                split=inner_split,
                x_cols=x_cols,
                cfg=cfg,
                variant=params_variant["variant"],
                sigma_init=float(params_variant["sigma_init"]),
                kappa_init=float(params_variant["kappa_init"]),
                xi_init=float(params_variant["xi_init"]),
                batch_size=int(params_variant["batch_size"]),
                censor_threshold=float(params_variant["censor_threshold"]),
                lr=float(params_variant["lr"]),
                n_ep=int(params_variant["n_ep"]),
                seed=seed,
                device=device,
                weight_decay=float(params_variant["weight_decay"]),
            )

            row = params_variant.copy()
            row.update(res)
            rows.append(row)

            print(
                f"tested x_set={x_set_name:18s} | "
                f"init={init_source:8s} | "
                f"variant={variant:10s} | "
                f"widths={params_variant['widths']} | "
                f"n_cov={len(x_cols):2d} | "
                f"valid_loss={res['valid_loss']:.4f} | "
                f"train_loss={res['train_loss']:.4f} | "
                f"stopped_epoch={res['stopped_epoch']}"
            )

    tuning_df = pd.DataFrame(rows)

    if len(tuning_df) == 0:
        raise RuntimeError("No NN configuration was successfully evaluated.")

    tuning_df = tuning_df.sort_values(
        [
            "valid_loss",
            "train_loss",
            "n_covariates",
            "stopped_epoch",
        ]
    ).reset_index(drop=True)

    best_params = tuning_df.iloc[0].to_dict()

    return tuning_df, best_params



def rerank_top_nn_configs(
    df_model: pd.DataFrame,
    outer_split: Dict[str, Any],
    x_sets: dict[str, list[str]],
    tuning_df: pd.DataFrame,
    top_k: int = 10,
    seed: int = 1,
    device: Optional[str] = None,
):
    df_outer_train = df_model.loc[outer_split["train_idx"]].copy()

    inner_split = make_single_split_from_train(
        df_outer_train,
        train_frac=0.8,
        block="30D",
        seed=seed + 100,
    )

    rows = []

    top_configs = tuning_df.head(top_k).copy()

    for _, params in top_configs.iterrows():
        params = params.to_dict()

        x_set_name = params["x_set_name"]
        x_cols = x_sets[x_set_name]

        cfg = Config(
            name=f"nn_rerank_{x_set_name}_{params['variant']}_{params['widths']}",
            widths=parse_widths(params["widths"]),
        )

        res = evaluate_nn_config_on_split(
            df_model=df_model,
            split=inner_split,
            x_cols=x_cols,
            cfg=cfg,
            variant=params["variant"],
            sigma_init=float(params["sigma_init"]),
            kappa_init=float(params["kappa_init"]),
            xi_init=float(params["xi_init"]),
            batch_size=int(params["batch_size"]),
            censor_threshold=float(params["censor_threshold"]),
            lr=float(params["lr"]),
            n_ep=int(params["n_ep"]),
            seed=seed,
            device=device,
            weight_decay=float(params.get("weight_decay", 0.0)),
        )

        row = params.copy()
        row.update(res)
        rows.append(row)

        print(
            f"reranked x_set={x_set_name:18s} | "
            f"init={params.get('init_source', 'unknown'):8s} | "
            f"variant={params['variant']:10s} | "
            f"valid_loss={res['valid_loss']:.4f} | "
            f"twCRPS_sum={res['twcrps_paper_sum']:.4f} | "
            f"sMAD={res['smad']:.4f} | "
            f"CRPS={res['crps_mean']:.4f} | "
            f"err95={res['err95']:.4f} | "
            f"err99={res['err99']:.4f}"
        )

    rerank_df = pd.DataFrame(rows)

    rerank_df["rank_valid_loss"] = rerank_df["valid_loss"].rank(method="average")
    rerank_df["rank_twcrps"] = rerank_df["twcrps_paper_sum"].rank(method="average")
    rerank_df["rank_smad"] = rerank_df["smad"].rank(method="average")
    rerank_df["rank_crps"] = rerank_df["crps_mean"].rank(method="average")
    rerank_df["rank_err95"] = rerank_df["err95"].rank(method="average")
    rerank_df["rank_err99"] = rerank_df["err99"].rank(method="average")

    rerank_df = rerank_df.sort_values(
        [
            "valid_loss",
            "twcrps_paper_sum",
            "crps_mean",
            "smad",
        ]
    ).reset_index(drop=True)

    best_params = rerank_df.iloc[0].to_dict()

    return rerank_df, best_params
