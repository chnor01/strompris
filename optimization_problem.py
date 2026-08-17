from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import pulp

from data_analysis import build_features, load_parquet
from train import CATEGORICAL_COLS, FEATURE_COLS, TARGET_COL


@dataclass
class ChargingPlan:
    """Result of the optimization: how much to charge each hour, and total cost"""

    hours: list[pd.Timestamp]
    kwh_per_hour: list[float]
    prices: list[float]
    total_cost: float
    total_kwh: float

    def summary(self) -> str:
        lines = [
            f"Charging plan - total {self.total_kwh:.1f} kWh, cost {self.total_cost:.2f} kr\n"
        ]
        for t, kwh, price in zip(self.hours, self.kwh_per_hour, self.prices):
            if kwh > 0.01:
                lines.append(
                    f"  {t.strftime('%Y-%m-%d %H:%M')}: {kwh:.2f} kWh @ {price:.3f} kr/kWh"
                )
        return "\n".join(lines)


def optimize_charging(
    price_forecast: pd.DataFrame,
    kwh_needed: float,
    max_kwh_per_hour: float = 11.0,
    timestamp_col: str = "timestamp",
    price_col: str = "pris_nok_kwh",
) -> ChargingPlan:
    """Find the plan that minimizes cost. Returns ChargingPlan with hourly distribution and total cost"""
    hours = price_forecast[timestamp_col].tolist()
    prices = price_forecast[price_col].tolist()
    n = len(hours)

    if n == 0:
        raise ValueError("Empty price forecast - no hours to optimize over")

    if kwh_needed > n * max_kwh_per_hour:
        raise ValueError(
            f"Impossible scenario: {kwh_needed} kWh cannot be charged in {n} hours "
            f"with a max of {max_kwh_per_hour} kWh/h (max possible: {n * max_kwh_per_hour} kWh)"
        )

    problem = pulp.LpProblem("ev_charging", pulp.LpMinimize)

    # one decision variable per hour: how many kwh to charge that hour
    charging = [
        pulp.LpVariable(f"charging_{i}", lowBound=0, upBound=max_kwh_per_hour)
        for i in range(n)
    ]

    # minimize total cost (price * kWh, summed over all hours)
    problem += pulp.lpSum(prices[i] * charging[i] for i in range(n))

    # constraint: total charging must meet the requirement exactly
    problem += pulp.lpSum(charging) == kwh_needed

    # Coin-or Branch and Cut
    problem.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[problem.status] != "Optimal":
        raise RuntimeError(
            f"No optimal solution found: {pulp.LpStatus[problem.status]}"
        )

    kwh_per_hour = [charging[i].varValue for i in range(n)]
    total_cost = sum(prices[i] * kwh_per_hour[i] for i in range(n))

    return ChargingPlan(
        hours=hours,
        kwh_per_hour=kwh_per_hour,
        prices=prices,
        total_cost=total_cost,
        total_kwh=sum(kwh_per_hour),
    )


def naive_charging(
    price_forecast: pd.DataFrame, kwh_needed: float, price_col="pris_nok_kwh"
):
    """Reference for comparison: charge at a constant rate from now on, regardless of price"""
    n = len(price_forecast)
    kwh_per_hour_naive = kwh_needed / n
    cost_naive = sum(price_forecast[price_col] * kwh_per_hour_naive)
    return cost_naive


def train_final_model(df: pd.DataFrame, params: dict) -> lgb.LGBMRegressor:
    """Train the model on all history (not walk-forward split)"""
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    model = lgb.LGBMRegressor(**params)
    model.fit(df[FEATURE_COLS], df[TARGET_COL], categorical_feature=CATEGORICAL_COLS)
    return model


def make_price_forecast(
    model: lgb.LGBMRegressor,
    history: pd.DataFrame,
    n_hours_ahead: int,
) -> pd.DataFrame:
    """Generates a price forecast n_hours_ahead into the future because lag features (pris_lag_24h etc.) for the later hours
    in the forecast window depend on predicted prices from earlier in the same window, not actual observed prices (which don't exist yet)"""
    history = history.sort_values("timestamp").reset_index(drop=True)
    last_known_time = history["timestamp"].max()

    working_data = history.copy()

    forecasts = []

    for i in range(1, n_hours_ahead + 1):
        new_time = last_known_time + pd.Timedelta(hours=i)

        new_row = _build_future_row(working_data, new_time)

        X = new_row[FEATURE_COLS].to_frame().T.astype(float)

        predicted_price = model.predict(X)[0]
        new_row[TARGET_COL] = predicted_price

        forecasts.append({"timestamp": new_time, "pris_nok_kwh": predicted_price})

        working_data = pd.concat(
            [working_data, new_row.to_frame().T], ignore_index=True
        )

    return pd.DataFrame(forecasts)


def _build_future_row(working_data: pd.DataFrame, new_time: pd.Timestamp) -> pd.Series:
    """Build one row of features for a future timestamp, based on working_data"""
    row = pd.Series(index=working_data.columns, dtype=object)
    row["timestamp"] = new_time

    price_series = working_data.set_index("timestamp")["pris_nok_kwh"]

    def get_price(hours_offset):
        t = new_time - pd.Timedelta(hours=hours_offset)
        return price_series.get(t, np.nan)

    row["pris_lag_24h"] = get_price(24)
    row["pris_lag_168h"] = get_price(168)
    row["pris_rullende_24h"] = np.mean([get_price(h) for h in range(1, 25)])
    row["pris_rullende_168h"] = np.mean([get_price(h) for h in range(1, 169)])

    last_known_weather = working_data.iloc[-1]
    for col in ["temperatur_c", "vind_m_s", "nedbor_mm_24h", "fyllingsgrad"]:
        row[col] = last_known_weather[col]

    temp_series = working_data.set_index("timestamp")["temperatur_c"]
    wind_series = working_data.set_index("timestamp")["vind_m_s"]

    def get_temp(hours_offset):
        t = new_time - pd.Timedelta(hours=hours_offset)
        return temp_series.get(t, last_known_weather["temperatur_c"])

    row["temp_rullende_24h"] = np.mean([get_temp(h) for h in range(24)])
    row["temp_rullende_72h"] = np.mean([get_temp(h) for h in range(72)])
    row["temp_min_72h"] = np.min([get_temp(h) for h in range(72)])
    row["temp_endring_24h"] = last_known_weather["temperatur_c"] - get_temp(24)
    row["vind_rullende_24h"] = np.mean(
        [
            wind_series.get(
                new_time - pd.Timedelta(hours=h), last_known_weather["vind_m_s"]
            )
            for h in range(24)
        ]
    )

    fill_level_series = working_data.set_index("timestamp")["fyllingsgrad"]
    fill_level_week_ago = fill_level_series.get(
        new_time - pd.Timedelta(hours=168), last_known_weather["fyllingsgrad"]
    )
    row["fyllingsgrad_endring_7d"] = (
        last_known_weather["fyllingsgrad"] - fill_level_week_ago
    )

    local = new_time.tz_convert("Europe/Oslo")
    row["time_pa_dognet"] = local.hour
    row["ukedag"] = local.dayofweek
    row["er_helg"] = int(local.dayofweek >= 5)
    row["maned"] = local.month

    return row


if __name__ == "__main__":
    df = load_parquet("data/NO1_historical.parquet")
    df = df[df["timestamp"] >= "2023-01-01"]
    df = build_features(df)

    best_params = pd.read_csv("data/best_lightgbm_params.csv").iloc[0].to_dict()
    best_params.update({"objective": "regression", "metric": "mae", "verbose": -1})

    for int_field in ["num_leaves", "n_estimators", "max_depth", "min_child_samples"]:
        best_params[int_field] = int(best_params[int_field])

    print("Training final model on all history...")
    model = train_final_model(df, best_params)

    print("Generating price forecast for the next 24 hours...")
    forecast = make_price_forecast(model, df, n_hours_ahead=24)
    print(forecast)

    plan = optimize_charging(forecast, kwh_needed=40.0, max_kwh_per_hour=11.0)
    print(f"\n{plan.summary()}")

    naive_cost = naive_charging(forecast, kwh_needed=40.0)
    print(f"\nNaive flat charging would cost: {naive_cost:.2f} kr")
    print(f"Savings: {naive_cost - plan.total_cost:.2f} kr")
