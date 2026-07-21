import httpx, pandas as pd, time, os, re
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv
from httpx_retries import Retry, RetryTransport


load_dotenv()


class StromprisClient:
    """Fetches prices for electricity from hvakosterstrommen.no"""
    
    BASE_URL = "https://www.hvakosterstrommen.no/api/v1/prices"
    AREAS = {"NO1", "NO2", "NO3", "NO4", "NO5"}
    
    def __init__(
        self, 
        retry_limit: int = 3, 
        backoff_factor: float = 0.5, 
        timeout: float = 30,
        area: str = "NO1", 
        data_dir: str | Path = "data/strompris"
        ):
        retry = Retry(total=retry_limit, backoff_factor=backoff_factor)
        transport = RetryTransport(retry=retry)
        self.client = httpx.Client(transport=transport, timeout=timeout)
        
        self.area = area 
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def _build_url(self, day: date):
        return f"{self.BASE_URL}/{day.year}/{day.month:02d}-{day.day:02d}_{self.area}.json"
        
    def get_day(self, day):
        """Get hourly prices for one day. Returns a dataframe"""
        data_path = self._data_path(day)
        if data_path.exists():
            return pd.read_parquet(data_path)
        
        url = self._build_url(day)
        data = self._get_response(url)
        
        df = self._convert_data(data)
        
        if not df.empty:
            df.to_parquet(data_path, index=False)
            
        return df
            
    def get_range(self, start: date, end: date):
        """Get hourly prices from a range of days"""
        dataframes = []
        current = start
        while current <= end:
            df = self.get_day(current)
            if not df.empty:
                dataframes.append(df)
            current += timedelta(days=1)
            
            
        if not dataframes:
            return pd.DataFrame()
        
        return pd.concat(dataframes, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        
    def _get_response(self, url):
        """HTTP response"""
        try:
            res = self.client.get(url)
            res.raise_for_status()
            time.sleep(1)
            return res.json()
        except httpx.HTTPError as exc:
            print(f"HTTP exception: {exc}")
            return None
    
    def _convert_data(self, data: list[dict]):
        """Convert data from json to dataframe"""
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["time_start"], utc=True)
        df["prisområde"] = self.area
        df = df.rename(columns={"NOK_per_kWh": "pris_nok_kwh", "EUR_per_kWh": "pris_eur_kwh"})
 
        return df[["timestamp", "prisområde", "pris_nok_kwh", "pris_eur_kwh"]]
    
    def _data_path(self, day: date):
        return self.data_dir / f"{self.area}_{day.isoformat()}.parquet"
    
    

class FrostClient:
    """Fetches weather observations from frost.met.no"""
    
    BASE_URL = "https://frost.met.no/observations/v0.jsonld"
    ELEMENTS = ["air_temperature", "wind_speed", "sum(precipitation_amount P1D)"]
 
    def __init__(
        self,
        client_id: str,
        retry_limit: int = 3,
        backoff_factor: float = 0.5,
        timeout: float = 30, 
        station_id: str = "SN18700",
        data_dir: str | Path = "data/frost",
        ):
        retry = Retry(total=retry_limit, backoff_factor=backoff_factor)
        transport = RetryTransport(retry=retry)
        self.client = httpx.Client(transport=transport, auth=(client_id, ""), timeout=timeout)
 
        self.station_id = station_id
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
 
    def _build_params_hourly(self, start: date, end: date):
        """For air temperature and wind speed. Hourly data"""
        
        # fixes timestamp mismatch between strompris/weather data
        padded_start = start - timedelta(days=1)
        padded_end = end + timedelta(days=1)
        return [
            {
                "sources": self.station_id,
                "elements": "air_temperature",
                "referencetime": f"{padded_start.isoformat()}T00:00:00Z/{padded_end.isoformat()}T00:00:00Z",
                "timeresolutions": "PT1H",
                "levels": "2",
            },
            {
                "sources": self.station_id,
                "elements": "wind_speed",
                "referencetime": f"{padded_start.isoformat()}T00:00:00Z/{padded_end.isoformat()}T00:00:00Z",
                "timeresolutions": "PT1H",
                "levels": "10",
            },
        ]

    def _build_params_precip(self, start: date, end: date):
        """For precipitation. Daily data"""
        padded_start = start - timedelta(days=1)
        padded_end = end + timedelta(days=1)
        return {
            "sources": self.station_id,
            "elements": "sum(precipitation_amount P1D)",
            "referencetime": f"{padded_start.isoformat()}T00:00:00Z/{padded_end.isoformat()}T00:00:00Z",
            "timeresolutions": "P1D",
            "timeoffsets": "PT6H",
        }
 
    def get_range(self, start: date, end: date):
        """Get weather observations for a date range. Returns a dataframe"""
        data_path = self._data_path(start, end)
        if data_path.exists():
            return pd.read_parquet(data_path)

        all_data = []
        for params in self._build_params_hourly(start, end):
            response = self._get_response(params)
            all_data += response or []

        precip_response = self._get_response(self._build_params_precip(start, end))
        all_data += precip_response or []

        df = self._convert_data(all_data)

        if not df.empty:
            df.to_parquet(data_path, index=False)

        return df
 
    def get_range_chunked(self, start: date, end: date, chunk_days: int = 90):
        """Get weather observations for a long date range, fetched in chunks"""
        dataframes = []
        current = start
        while current < end:
            chunk_end = min(current + pd.Timedelta(days=chunk_days), end)
            df = self.get_range(current, chunk_end)
            if not df.empty:
                dataframes.append(df)
            current = chunk_end
 
        if not dataframes:
            return pd.DataFrame()
 
        return (
            pd.concat(dataframes, ignore_index=True)
            .drop_duplicates(subset=["valid_time"], keep="first")
            .sort_values("valid_time")
            .reset_index(drop=True)
            )
 
    def _get_response(self, params):
        """HTTP response"""
        try:
            res = self.client.get(self.BASE_URL, params=params)
            res.raise_for_status()
            time.sleep(1)
            return res.json().get("data", [])
        except httpx.HTTPError as exc:
            print(f"HTTP exception: {exc}")
            return None
 
    def _convert_data(self, data: list[dict]):
        """Convert data from json to dataframe"""
        if not data:
            return pd.DataFrame()
 
        rows = []
        for entry in data:
            reference_time = pd.to_datetime(entry["referenceTime"], utc=True)
            for obs in entry.get("observations", []):
                offset = self._parse_offset(obs.get("timeOffset", "PT0H"))
                rows.append({
                    "valid_time": reference_time + offset,
                    "element_id": obs["elementId"],
                    "value": obs["value"],
                })
 
        if not rows:
            return pd.DataFrame()
 
        long_df = pd.DataFrame(rows)
        wide_df = long_df.pivot_table(
            index="valid_time", columns="element_id", values="value", aggfunc="first"
        ).reset_index()
        wide_df.columns.name = None
 
        return wide_df.rename(columns={
            "air_temperature": "temperatur_c",
            "wind_speed": "vind_m_s",
            "sum(precipitation_amount P1D)": "nedbor_mm_24h",
        })
 
    @staticmethod
    def _parse_offset(offset: str):
        match = re.fullmatch(r"PT(\d+)H", offset)
        return pd.Timedelta(hours=int(match.group(1))) if match else pd.Timedelta(hours=0)
 
    def _data_path(self, start: date, end: date):
        return self.data_dir / f"{self.station_id}_{start.isoformat()}_{end.isoformat()}.parquet"


def merge_data(strompriser: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Combine electricity prices and weather data to one format: timestamp, prisområde, pris_nok_kwh, temperatur_c, vind_m_s, nedbor_mm_24h"""
    df = strompriser.merge(
        weather,
        left_on="timestamp",
        right_on="valid_time",
        how="left",
    )
 
    df["nedbor_mm_24h"] = df["nedbor_mm_24h"].ffill()
 
    df = df.drop(columns=["valid_time"])
 
    return df.sort_values("timestamp").reset_index(drop=True)



strompris = StromprisClient()

client_id = os.getenv("FROST_CLIENT_ID")
frost = FrostClient(client_id=client_id)


def merge_backfill(area: str = "NO1"):
    """Fetch strompriser and weather data for a larger time range, merge and save"""
    # 2022 earliest data for hvakosterstrommen
    start = date(2026, 1, 1)
    end = date.today() - timedelta(days=1)

    strompriser = strompris.get_range(start, end)
    weather = frost.get_range_chunked(start, end, chunk_days=90)
    
    if strompriser.empty:
        print("No strompriser!")
        return None
    
    if weather.empty or weather[["temperatur_c", "vind_m_s"]].isna().any().any():
        print("Weather data missing values, but saving...")

    merged_backfill = merge_data(strompriser=strompriser, weather=weather)
    
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{area}_historical.parquet"
    
    merged_backfill.to_parquet(output_path, index=False)
    print(f"Saved {len(merged_backfill)} rows for time range: ( {start} to {end} ) -> {output_path}")
    
    return merged_backfill


def fetch_day(target_date: date | None = None, area: str = "NO1") -> pd.DataFrame:
    """Fetch strompris and weather data for one data, merge and save. Defaults to yesterday's data if target_date not specified"""
    if target_date is None:
        target_date = date.today() - timedelta(days=1)
    
    strom_today = strompris.get_day(target_date)
    weather_today = frost.get_range(target_date, target_date + timedelta(days=1))
    
    if strom_today.empty:
        print("No strompriser!")
        return None
    
    if weather_today.empty or weather_today[["temperatur_c", "vind_m_s"]].isna().any().any():
        print("Weather data missing values, but saving...")
    
    merged_today = merge_data(strom_today, weather_today)
    print(merged_today.head(30))
    
    output_dir = Path("data/daily")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{area}_{target_date.isoformat()}.parquet"
    
    merged_today.to_parquet(output_path, index=False)
    print(f"Saved {len(merged_today)} rows for {target_date} -> {output_path}")
    
    return merged_today
    

def load_parquet(path: str | Path = "data/NO1_historical.parquet") -> pd.DataFrame:
    """Load parquet data"""
    data_path = Path(path)
    if not data_path.exists():
        return None
    
    df = pd.read_parquet(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    return df.sort_values("timestamp").reset_index(drop=True)


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
    print(f"\nAntall unike duplikate timestamps: {duplicates['timestamp'].nunique()}")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prices at t-24h (1 day) and t-168h (1 week), rolling average, and time columns"""
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



#merge_backfill()
#fetch_day()

#data = load_parquet()
#print(average_prices(data))
#print(negative_and_spike_periods(data))
#print(missing_data(data))
#features = build_features(data)
#save_features(features)

# todo: find corresponding weather stations to strompris areas

