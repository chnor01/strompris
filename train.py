import pandas as pd, numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dataclasses import dataclass
from data_analysis import load_parquet
from statsmodels.tsa.statespace.sarimax import SARIMAX

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
            splits.append(Split(
                fold=fold,
                train=train,
                test=test,
                train_start=data_start,
                train_end=test_start,
                test_start=test_start,
                test_end=test_end,
                ))

        test_start += pd.Timedelta(days=step_days)

    return splits

def print_splits(splits: list[Split]):
    print("num folds", len(splits))
    for s in splits:
        print(f"train {s.train_start.date()} -> {s.train_end.date()} ({len(s.train)} rows) === test {s.test_start.date()} -> {s.test_end.date()} ({len(s.test)} rows)")
        overlap = s.train["timestamp"].max() >= s.test["timestamp"].min()
        assert not overlap, f"Fold {s.fold}: overlaps"
    print("\nno overlaps")
    
    
def seasonal_naive_predict(test: pd.DataFrame, lag_col: str = "pris_lag_24h") -> pd.Series:
    return test[lag_col]
 
 
def evaluate_baseline(splits: list[Split], target_col: str = "pris_nok_kwh") -> pd.DataFrame:
    """Evaluate seasonal-naive (t-24 and t-168) on each fold"""
    results = []
 
    for split in splits:
        test = split.test.dropna(subset=["pris_lag_24h", "pris_lag_168h", target_col])
 
        if test.empty:
            print(f"Fold {split.fold}: no valid test rows (missing lag values), skipping")
            continue
 
        for lag_col, name in [("pris_lag_24h", "naive_t-24"), ("pris_lag_168h", "naive_t-168")]:
            y_true = test[target_col]
            y_pred = seasonal_naive_predict(test, lag_col)
 
            results.append({
                "fold": split.fold,
                "model": name,
                "mae": mean_absolute_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "n": len(test),
            })
 
    return pd.DataFrame(results)
 
 
def train_sarima(train: pd.DataFrame, test: pd.DataFrame, target_col: str = "pris_nok_kwh",
                  order=(1, 1, 1), seasonal_order=(1, 0, 1, 24)):
    
    y_train = train.set_index("timestamp")[target_col].asfreq("h")
    y_train = y_train.ffill()  # SARIMA doesn't tolerate gaps in the time series
 
    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
 
    forecast = fit.forecast(steps=len(test))
    return forecast.values
 
 
def evaluate_sarima(splits: list[Split], target_col: str = "pris_nok_kwh") -> pd.DataFrame:
    """Evaluate SARIMA on each fold"""
    results = []
 
    for split in splits:
        train = split.train
        
 
        test = split.test.dropna(subset=[target_col])
        if test.empty:
            continue
 
        print(f"Fold {split.fold}: training SARIMA on {len(train)} rows...")
        try:
            y_pred = train_sarima(train, test, target_col=target_col)
        except Exception as exc:
            print(f"Fold {split.fold}: SARIMA failed ({exc}), skipping")
            continue
 
        y_true = test[target_col].values
 
        results.append({
            "fold": split.fold,
            "model": "sarima",
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "n": len(test),
        })
 
    return pd.DataFrame(results)
    


if __name__ == "__main__":
    df = load_parquet("data/NO1_features.parquet")
    splits = create_splits(df, test_size_days=7, min_train_days=180, step_days=30)
    #print_splits(splits)
    print(len(splits))
    
    def baseline():
        naive_results = evaluate_baseline(splits)
        print(naive_results)
        print("\nAverage per model:")
        print(naive_results.groupby("model")[["mae", "rmse"]].mean())
    
    def sarimax():
        sarima_results = evaluate_sarima(splits)
        print(sarima_results)
        if not sarima_results.empty:
            print("\nAverage:")
            print(sarima_results[["mae", "rmse"]].mean())
            
    #print(baseline())
    print(sarimax())