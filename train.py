from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_analysis import build_features, load_parquet


@dataclass
class Split:
    fold: int
    train: pd.DataFrame
    test: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def create_splits(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    test_size_days: int = 30,
    min_train_days: int = 90,
    step_days: int | None = None,
) -> list[Split]:
    """create train/test splits, no overlapping time windows. train data walks forward a number of days every fold.
    test data is always in chronological order after train data.
    ex: train 2025-12-31 -> 2026-03-01 (1440 rows) === test 2026-03-01 -> 2026-03-31 (720 rows)
        train 2025-12-31 -> 2026-03-15 (1776 rows) === test 2026-03-15 -> 2026-04-14 (720 rows)
        train 2025-12-31 -> 2026-03-29 (2112 rows) === test 2026-03-29 -> 2026-04-28 (720 rows)
    """
    if step_days is None:
        step_days = test_size_days

    df = df.sort_values(timestamp_col).reset_index(drop=True)

    data_start = df[timestamp_col].min()
    data_end = df[timestamp_col].max()

    splits = []
    fold = 0
    test_start = data_start + pd.Timedelta(days=min_train_days)

    while test_start + pd.Timedelta(days=test_size_days) <= data_end:
        test_end = test_start + pd.Timedelta(days=test_size_days)

        train = df[df[timestamp_col] < test_start]
        test = df[(df[timestamp_col] >= test_start) & (df[timestamp_col] < test_end)]

        if not train.empty and not test.empty:
            fold += 1
            splits.append(
                Split(
                    fold=fold,
                    train=train,
                    test=test,
                    train_start=data_start,
                    train_end=test_start,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

        test_start += pd.Timedelta(days=step_days)

    return splits


def print_splits(splits: list[Split]):
    print("num folds", len(splits))
    for s in splits:
        print(
            f"train {s.train_start.date()} -> {s.train_end.date()} ({len(s.train)} rows) === test {s.test_start.date()} -> {s.test_end.date()} ({len(s.test)} rows)"
        )
        overlap = s.train["timestamp"].max() >= s.test["timestamp"].min()
        assert not overlap, f"Fold {s.fold}: overlaps"
    print("\nno overlaps")


def seasonal_naive_predict(
    test: pd.DataFrame, lag_col: str = "pris_lag_24h"
) -> pd.Series:
    return test[lag_col]


def evaluate_baseline(
    splits: list[Split], target_col: str = "pris_nok_kwh"
) -> pd.DataFrame:
    """Evaluate seasonal-naive (t-24 and t-168) on each fold"""
    results = []

    for split in splits:
        test = split.test.dropna(subset=["pris_lag_24h", "pris_lag_168h", target_col])

        if test.empty:
            print(
                f"Fold {split.fold}: no valid test rows (missing lag values), skipping"
            )
            continue

        for lag_col, name in [
            ("pris_lag_24h", "naive_t-24"),
            ("pris_lag_168h", "naive_t-168"),
        ]:
            y_true = test[target_col]
            y_pred = seasonal_naive_predict(test, lag_col)

            results.append(
                {
                    "fold": split.fold,
                    "model": name,
                    "mae": mean_absolute_error(y_true, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                    "n": len(test),
                }
            )

    return pd.DataFrame(results)


FEATURE_COLS = [
    "pris_lag_24h",
    "pris_lag_168h",
    "pris_rullende_24h",
    "pris_rullende_168h",
    "temperatur_c",
    "temp_rullende_24h",
    "temp_rullende_72h",
    "temp_min_72h",
    "temp_endring_24h",
    "vind_m_s",
    "vind_rullende_24h",
    "nedbor_mm_24h",
    "fyllingsgrad",
    "fyllingsgrad_endring_7d",
    "time_pa_dognet",
    "ukedag",
    "er_helg",
    "maned",
]
CATEGORICAL_COLS = ["ukedag", "maned"]

TARGET_COL = "pris_nok_kwh"


def train_lightgbm(train: pd.DataFrame, test: pd.DataFrame, params: dict | None = None):
    """Train LightGBM model"""
    if params is None:
        params = {
            "objective": "regression",
            "metric": "mae",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "verbose": -1,
        }

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test = test[FEATURE_COLS]

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL_COLS)

    y_pred = model.predict(X_test)
    return y_pred, model


def evaluate_lightgbm(splits: list[Split], params: dict | None = None) -> pd.DataFrame:
    """Evaluate LightGBM on every fold"""
    results = []
    models = []

    for split in splits:
        train = split.train.dropna(subset=FEATURE_COLS + [TARGET_COL])
        test = split.test.dropna(subset=FEATURE_COLS + [TARGET_COL])

        if train.empty or test.empty:
            print(f"Fold {split.fold}: empty/missing values, skipping...")
            continue

        y_pred, model = train_lightgbm(train, test, params=params)
        y_test = test[TARGET_COL].values

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append(
            {
                "fold": split.fold,
                "model": "lightgbm",
                "mae": mae,
                "rmse": rmse,
                "n": len(test),
            }
        )
        models.append(model)

        print(f"Fold {split.fold}: MAE={mae:.4f}, RMSE={rmse:.4f} (n={len(test)})")

    return pd.DataFrame(results), models


def print_feature_importance(model, top_n: int = 15):
    importance = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(f"Top {top_n} most important features")
    print(importance.head(top_n).to_string(index=False))


def objective(trial: optuna.Trial, splits) -> float:
    """Optuna suggests hyperparameters based on each trial's MAE"""
    params = {
        "objective": "regression",
        "metric": "mae",
        "verbose": -1,
        "num_leaves": trial.suggest_int("num_leaves", 15, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 150),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    results, _ = evaluate_lightgbm(splits, params=params)

    if results.empty:
        return float("inf")

    return float(results["mae"].median())


def optuna_search(splits: list[Split], n_trials: int = 50) -> optuna.Study:
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, splits),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\nBest median MAE: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")

    return study


if __name__ == "__main__":
    df = load_parquet("data/NO1_historical.parquet")
    df = df[df["timestamp"] >= "2023-01-01"]
    df = build_features(df)

    splits = create_splits(df, test_size_days=7, min_train_days=180, step_days=30)
    print(f"Number of splits: {len(splits)}")

    study = optuna_search(splits=splits, n_trials=100)

    best_params = {
        "objective": "regression",
        "metric": "mae",
        "verbose": -1,
        **study.best_params,
    }

    results, _ = evaluate_lightgbm(splits, params=best_params)

    print("LightGBM results with best params")
    print(f"Median MAE: {results['mae'].median():.4f}")
    print(f"Average MAE: {results['mae'].mean():.4f}")

    print("\nSeasonal-naive (t-24h and t-168h)")
    naive_results = evaluate_baseline(splits)
    print("Average MAE and RMSE:")
    print(naive_results.groupby("model")[["mae", "rmse"]].mean())

    naive_mae = naive_results[naive_results["model"] == "naive_t-24"]["mae"].median()
    improvement = (naive_mae - results["mae"].median()) / naive_mae
    print(f"\nSeasonal-naive (t-24) median MAE: {naive_mae:.4f}")
    print(f"Improvement (median): {improvement:.1%}")

    pd.DataFrame([study.best_params]).to_csv(
        "data/best_lightgbm_params.csv", index=False
    )
    print("Saved best params to data/best_lightgbm_params.csv")
