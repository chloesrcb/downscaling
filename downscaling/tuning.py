from itertools import product
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from downscaling.config import SIGMA_INIT, KAPPA_INIT
from downscaling.splits import make_single_split_from_train
from downscaling.nn import Config, parse_widths
from downscaling.evaluation import (
    evaluate_nn_config_on_split,
    evaluate_nn_config_on_split_fast,
)
from downscaling.regression import get_gam_initial_values
from downscaling.utils import make_param_grid
from downscaling.scores import score_one_prediction_table
from downscaling.prediction import fit_predict_nn_test

from downscaling.paths import OUT_DIR
from downscaling.config import (
    LAMBDA_PROP_KAPPA_GT2,
    LAMBDA_EXCESS_KAPPA,
    QUANTILES_FOR_DIAGNOSTICS,
)


def tune_nn_on_outer_train(
    df_model: pd.DataFrame,
    outer_split: Dict[str, Any],
    x_sets: dict[str, list[str]],
    param_grid: dict,
    seed: int = 1,
    device: Optional[str] = None,
    single_cov_col: Optional[str] = None,
):
    df_outer_train = df_model.loc[outer_split["train_idx"]].copy()

    inner_split = make_single_split_from_train(
        df_outer_train,
        train_frac=0.8,
        block="15D",
        seed=seed + 100,
    )

    rows = []
    histories = {}

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
        "kappa_max_nn",
        "lambda_kappa",
    ]

    common_values = [param_grid[k] for k in common_keys]
    config_id = 0

    for vals in product(*common_values):
        params = dict(zip(common_keys, vals))

        variant = params["variant"]
        x_set_name = params["x_set_name"]
        init_source = params["init_source"]

        if x_set_name not in x_sets:
            raise ValueError(f"Unknown x_set_name: {x_set_name}")

        x_cols = x_sets[x_set_name]

        xi_init_requested = float(params["xi_init"])
        xi_init_effective = xi_init_requested

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
                    xi_fixed=xi_init_requested,
                    sigma_init=SIGMA_INIT,
                    kappa_init=KAPPA_INIT,
                    censor_threshold=float(params["censor_threshold"]),
                )

                sigma_candidates = [gam_init["sigma_init_gam"]]
                kappa_candidates = [gam_init["kappa_init_gam"]]

                if "xi_init_gam" in gam_init:
                    xi_init_effective = float(gam_init["xi_init_gam"])
                elif "xi_hat" in gam_init:
                    xi_init_effective = float(gam_init["xi_hat"])
                elif "xi_fixed" in gam_init:
                    xi_init_effective = float(gam_init["xi_fixed"])
                else:
                    xi_init_effective = xi_init_requested

            except Exception as e:
                print(f"GAM init failed for {params}: {e}")
                continue

        for sigma_init, kappa_init in product(sigma_candidates, kappa_candidates):
            config_id += 1

            params_variant = params.copy()

            params_variant["config_id"] = config_id
            params_variant["sigma_init"] = float(sigma_init)
            params_variant["kappa_init"] = float(kappa_init)
            params_variant["xi_init_requested"] = xi_init_requested
            params_variant["xi_init_effective"] = xi_init_effective
            params_variant["xi_init"] = xi_init_effective
            params_variant["n_covariates"] = len(x_cols)

            if gam_init is not None:
                params_variant.update(gam_init)

            cfg = Config(
                name=(
                    f"nn_{config_id}_{x_set_name}_{variant}_"
                    f"{params_variant['widths']}_{init_source}_"
                    f"xi{xi_init_effective}"
                ),
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
                xi_init=float(params_variant["xi_init_effective"]),
                batch_size=int(params_variant["batch_size"]),
                censor_threshold=float(params_variant["censor_threshold"]),
                lr=float(params_variant["lr"]),
                n_ep=int(params_variant["n_ep"]),
                seed=seed,
                device=device,
                weight_decay=float(params_variant["weight_decay"]),
                return_history=True,
                kappa_max_nn=float(params_variant["kappa_max_nn"]),
                lambda_kappa=float(params_variant["lambda_kappa"]),
            )

            history = res.get("history", None)

            if history is not None:
                histories[config_id] = history

            row = params_variant.copy()
            row.update({k: v for k, v in res.items() if k != "history"})
            rows.append(row)

            print(
                f"tested config_id={config_id:03d} | "
                f"x_set={x_set_name:18s} | "
                f"init={init_source:8s} | "
                f"variant={variant:10s} | "
                f"widths={params_variant['widths']} | "
                f"n_cov={len(x_cols):2d} | "
                f"xi_requested={xi_init_requested:.4f} | "
                f"xi_effective={xi_init_effective:.4f} | "
                f"censor={float(params_variant['censor_threshold']):.3f} | "
                f"valid_loss={res['valid_loss']:.4f} | "
                f"train_loss={res['train_loss']:.4f} | "
                f"best_epoch={res['best_epoch']:.4f} | "
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
            "best_epoch",
            "stopped_epoch",
        ]
    ).reset_index(drop=True)

    best_params = tuning_df.iloc[0].to_dict()
    best_config_id = int(best_params["config_id"])
    best_history = histories.get(best_config_id, None)

    return tuning_df, best_params, best_history


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
        block="15D",
        seed=seed + 100,
    )

    rows = []
    histories = {}

    top_configs = tuning_df.head(top_k).copy()

    for rerank_id, (_, params) in enumerate(top_configs.iterrows(), start=1):
        params = params.to_dict()

        x_set_name = params["x_set_name"]
        x_cols = x_sets[x_set_name]

        cfg = Config(
            name=f"nn_rerank_{rerank_id}_{x_set_name}_{params['variant']}_{params['widths']}",
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
            return_history=True,
            kappa_max_nn=float(params["kappa_max_nn"]),
            lambda_kappa=float(params["lambda_kappa"]),
        )

        history = res.get("history", None)

        if history is not None:
            histories[rerank_id] = history

        row = params.copy()
        row["rerank_id"] = rerank_id
        row.update({k: v for k, v in res.items() if k != "history"})
        rows.append(row)

        print(
            f"reranked rerank_id={rerank_id:03d} | "
            f"x_set={x_set_name:18s} | "
            f"init={params.get('init_source', 'unknown'):8s} | "
            f"variant={params['variant']:10s} | "
            f"valid_loss={res['valid_loss']:.4f} | "
            f"twCRPS_sum={res['twcrps_sum']:.4f} | "
            f"sMAD={res['smad']:.4f} | "
            f"CRPS={res['crps_mean']:.4f} | "
            f"err95={res['err95']:.4f} | "
            f"err99={res['err99']:.4f}"
        )

    rerank_df = pd.DataFrame(rows)

    rerank_df["rank_valid_loss"] = rerank_df["valid_loss"].rank(method="average")
    rerank_df["rank_twcrps"] = rerank_df["twcrps_sum"].rank(method="average")
    rerank_df["rank_smad"] = rerank_df["smad"].rank(method="average")
    rerank_df["rank_crps"] = rerank_df["crps_mean"].rank(method="average")
    rerank_df["rank_err95"] = rerank_df["err95"].rank(method="average")
    rerank_df["rank_err99"] = rerank_df["err99"].rank(method="average")

    rerank_df = rerank_df.sort_values(
        [
            "twcrps_sum",
            "valid_loss",
            "crps_mean",
            "smad",
        ]
    ).reset_index(drop=True)

    best_params = rerank_df.iloc[0].to_dict()
    best_rerank_id = int(best_params["rerank_id"])
    best_history = histories.get(best_rerank_id, None)

    return rerank_df, best_params, best_history



# Tuning and final LOSO fitting
def select_tuning_stations(
    df: pd.DataFrame,
    stations: list[str],
    station_col: str,
    n_tuning_stations: int = 5,
) -> list[str]:
    station_summary = (
        df.groupby(station_col)
        .agg(
            n=("Y_obs", "size"),
            rain_freq=("Y_obs", lambda x: float(np.mean(x > 0))),
            y_q95=("Y_obs", lambda x: float(np.nanquantile(x, 0.95))),
            y_max=("Y_obs", "max"),
        )
        .reset_index()
    )

    station_summary = station_summary[station_summary[station_col].isin(stations)].copy()
    station_summary = station_summary.sort_values("rain_freq")
    station_summary.to_csv(OUT_DIR / "station_summary_for_tuning_selection.csv", index=False)

    if len(station_summary) <= n_tuning_stations:
        return station_summary[station_col].tolist()

    idx = np.linspace(0, len(station_summary) - 1, n_tuning_stations).round().astype(int)
    return station_summary.iloc[idx][station_col].tolist()


def tune_nn_loso(
    df_model: pd.DataFrame,
    stations_for_tuning: list[str],
    station_col: str,
    x_sets: dict,
    param_grid: dict,
    seed: int = 2026,
    device=None,
) -> tuple[pd.DataFrame, dict]:
    configs = make_param_grid(param_grid)
    rows = []

    print("\nStarting LOSO hyperparameter tuning")
    print("Tuning stations:", stations_for_tuning)
    print("Number of NN configurations:", len(configs))

    for cfg_id, params in enumerate(configs, start=1):
        print(f"\nConfig {cfg_id}/{len(configs)}")
        print(params)

        preds_cfg = []
        failed = False

        for j, station in enumerate(stations_for_tuning, start=1):
            print(f"  Tuning fold {j}/{len(stations_for_tuning)}: left-out station = {station}")

            df_train = df_model[df_model[station_col] != station].copy()
            df_valid = df_model[df_model[station_col] == station].copy()

            if len(df_valid) < 20:
                print(f"  Skipping {station}: too few observations.")
                continue

            try:
                _row, pred, _fit = fit_predict_nn_test(
                    df_train_valid=df_train,
                    df_test=df_valid,
                    x_sets=x_sets,
                    best_params=params,
                    seed=seed + 1000 * cfg_id + j,
                    device=device,
                )
            except Exception as e:
                print(f"  Failed for station {station}: {e}")
                failed = True
                break

            pred["model"] = "NN"
            pred["left_out_station"] = station
            preds_cfg.append(pred)

        if failed or len(preds_cfg) == 0:
            rows.append({"config_id": cfg_id, **params, "failed": True, "selection_score": np.inf})
            continue

        pred_cfg_all = pd.concat(preds_cfg, ignore_index=True, sort=False)
        pred_cfg_all = add_prediction_quantities(pred_cfg_all, quantiles=QUANTILES_FOR_DIAGNOSTICS)

        scores = score_one_prediction_table(pred_cfg_all, alpha=1.0)
        selection_score = (
            scores["twcrps_sum"]
            + LAMBDA_PROP_KAPPA_GT2 * scores["prop_kappa_gt_2"]
            + LAMBDA_EXCESS_KAPPA * scores["kappa_excess_mean"]
        )

        rows.append({
            "config_id": cfg_id,
            **params,
            "failed": False,
            **scores,
            "selection_score": selection_score,
        })

        print(
            f"  selection_score={selection_score:.6g} | "
            f"twCRPS_sum={scores['twcrps_sum']:.6g} | "
            f"twCRPS_mean={scores['twcrps_mean']:.6g} | "
            f"kappa_q99={scores['kappa_q99']:.3f} | "
            f"prop(kappa>2)={scores['prop_kappa_gt_2']:.3f}"
        )

    tuning_df = pd.DataFrame(rows).sort_values("selection_score").reset_index(drop=True)
    best_row = tuning_df.iloc[0].to_dict()
    best_params = {k: best_row[k] for k in param_grid.keys()}

    if isinstance(best_params.get("widths"), str):
        best_params["widths"] = eval(best_params["widths"])

    return tuning_df, best_params