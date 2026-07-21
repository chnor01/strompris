import pandas as pd
from dataclasses import dataclass
from build_dataset import load_parquet

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


df = load_parquet("data/NO1_features.parquet")
splits = create_splits(df, test_size_days=30, min_train_days=90, step_days=14)

print("num folds", len(splits))
for s in splits:
    print(f"train {s.train_start.date()} -> {s.train_end.date()} ({len(s.train)} rows) === test {s.test_start.date()} -> {s.test_end.date()} ({len(s.test)} rows)")
    overlap = s.train["timestamp"].max() >= s.test["timestamp"].min()
    assert not overlap, f"Fold {s.fold}: overlaps"
print("\nno overlaps")