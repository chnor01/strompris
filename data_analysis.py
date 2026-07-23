import pandas as pd
from pathlib import Path

def average_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Average stromprices per hour, day, month"""
    df = df.copy()
 
    local = df["timestamp"].dt.tz_convert("Europe/Oslo")
    df["time"] = local.dt.hour
    df["ukedag"] = local.dt.day_name()
    df["maned"] = local.dt.month
    df["er_helg"] = local.dt.dayofweek >= 5
    
    per_time = df.groupby("time")["pris_nok_kwh"].agg(["mean", "std", "count"])
    per_ukedag = df.groupby("ukedag")["pris_nok_kwh"].agg(["mean", "std", "count"])
    per_maned = df.groupby("maned")["pris_nok_kwh"].agg(["mean", "std", "count"])
    
    print("=== Prices per hour ===")
    print(per_time)
    print("\n=== Prices per day ===")
    print(per_ukedag)
    print("\n=== Prices per month ===")
    print(per_maned)
    
    return df


def negative_and_spike_periods(df: pd.DataFrame, spike_quantile: float = 0.99) -> dict:
    negative = df[df["pris_nok_kwh"] < 0]
    spike_threshold = df["pris_nok_kwh"].quantile(spike_quantile)
    spikes = df[df["pris_nok_kwh"] > spike_threshold]
 
    print(f"Hours w/ negative prices: {len(negative)} ({len(negative) / len(df):.2%})")
    if not negative.empty:
        print(negative[["timestamp", "pris_nok_kwh"]].describe())
 
    print(f"\n=== Spike-periods (>{spike_quantile:.0%}-quantile = {spike_threshold:.2f} kr/kWh) ===")
    print(f"Number of hours: {len(spikes)} ({len(spikes) / len(df):.2%})")
 
    return {"negative": negative, "spikes": spikes, "spike_threshold": spike_threshold}

def missing_data(df: pd.DataFrame):
    print("\n=== Missing values ===")
    print(df.isna().sum())
 
    print(f"From: {df['timestamp'].min()}")
    print(f"To: {df['timestamp'].max()}")
    expected_rows = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600 + 1
    print(f"Expected number of hours: {expected_rows:.0f}, Actual: {len(df)}")

    duplicates = df[df.duplicated(subset=["timestamp"], keep=False)]
    print(duplicates.sort_values("timestamp"))
    print(f"\nNumber of unique duplicate timestamps: {duplicates['timestamp'].nunique()}")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add more columns for prices at t-24h (1 day) and t-168h (1 week), rolling average, and time info"""
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
 
    df["pris_lag_24h"] = df["pris_nok_kwh"].shift(24)
    df["pris_lag_168h"] = df["pris_nok_kwh"].shift(168)
 
    df["pris_rullende_24h"] = df["pris_nok_kwh"].shift(1).rolling(24).mean()
    df["pris_rullende_168h"] = df["pris_nok_kwh"].shift(1).rolling(168).mean()

    local = df["timestamp"].dt.tz_convert("Europe/Oslo")
    df["time_pa_dognet"] = local.dt.hour
    df["ukedag"] = local.dt.dayofweek
    df["er_helg"] = (local.dt.dayofweek >= 5).astype(int)
    df["maned"] = local.dt.month

    decimal_columns = [
    "pris_lag_24h",
    "pris_lag_168h",
    "pris_rullende_24h",
    "pris_rullende_168h",
    ]
    df[decimal_columns] = df[decimal_columns].round(5)

    return df

def save_features(df: pd.DataFrame, area: str = "NO1"):
    output_path = Path(f"data/{area}_features.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
    
    
def load_parquet(path: str | Path = "data/NO1_historical.parquet") -> pd.DataFrame:
    """Load parquet data"""
    data_path = Path(path)
    if not data_path.exists():
        return None
    
    df = pd.read_parquet(data_path)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    return df.sort_values("timestamp").reset_index(drop=True)
    
if __name__ == "__main__":
    data = load_parquet()
    print(average_prices(data))
    print(negative_and_spike_periods(data))
    print(missing_data(data))

    features = build_features(data)
    save_features(features)