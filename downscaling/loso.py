
from typing import Dict, Any
from downscaling.settings import (
    DEVICE,
    QUANTILES_FOR_DIAGNOSTICS,
    REFERENCE_MODEL,
    SEED,)
from downscaling.prediction import (
    fit_predict_nn_test,
    fit_predict_regression_test,
    fit_predict_stationary_test,
)
from downscaling.diagnostics import add_prediction_quantities

import pandas as pd



def run_final_loso_evaluation(
    df_model: pd.DataFrame,
    stations: list[str],
    station_col: str,
    x_sets: dict,
    best_params_final: dict,
    single_cov_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_preds = []
    all_native_rows = []

    for i, station in enumerate(stations, start=1):
        print(f"\n[{i}/{len(stations)}] Leaving out station: {station}")

        df_train = df_model[df_model[station_col] != station].copy()
        df_test = df_model[df_model[station_col] == station].copy()

        if len(df_test) < 20:
            print(f"Skipping {station}: too few observations.")
            continue

        print("Train n:", len(df_train), "| Test n:", len(df_test))

        row, pred = fit_predict_stationary_test(df_train_valid=df_train, df_test=df_test)
        row["model"] = REFERENCE_MODEL
        row["left_out_station"] = station
        pred["model"] = REFERENCE_MODEL
        pred["left_out_station"] = station
        all_native_rows.append(row)
        all_preds.append(pred)

        row, pred = fit_predict_regression_test(
            df_train_valid=df_train,
            df_test=df_test,
            model_type="glm",
            variant="both",
            covariate_col=single_cov_col,
        )
        row["model"] = "GLM"
        row["left_out_station"] = station
        pred["model"] = "GLM"
        pred["left_out_station"] = station
        all_native_rows.append(row)
        all_preds.append(pred)

        row, pred = fit_predict_regression_test(
            df_train_valid=df_train,
            df_test=df_test,
            model_type="gam",
            variant="both",
            covariate_col=single_cov_col,
        )
        row["model"] = "GAM"
        row["left_out_station"] = station
        pred["model"] = "GAM"
        pred["left_out_station"] = station
        all_native_rows.append(row)
        all_preds.append(pred)

        row, pred, _fit = fit_predict_nn_test(
            df_train_valid=df_train,
            df_test=df_test,
            x_sets=x_sets,
            best_params=best_params_final,
            seed=SEED + i,
            device=DEVICE,
        )
        row["model"] = "NN"
        row["left_out_station"] = station
        pred["model"] = "NN"
        pred["left_out_station"] = station
        all_native_rows.append(row)
        all_preds.append(pred)

    native_rows = pd.DataFrame(all_native_rows)
    pred_loso_all = pd.concat(all_preds, ignore_index=True, sort=False)
    pred_loso_all = add_prediction_quantities(pred_loso_all, quantiles=QUANTILES_FOR_DIAGNOSTICS)

    return native_rows, pred_loso_all
