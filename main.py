import httpx, pandas as pd, time, os, json, re
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
        return {
            "sources": self.station_id,
            "elements": "air_temperature,wind_speed",
            "referencetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T00:00:00Z",
            "timeresolutions": "PT1H",
            "levels": "2,10",
        }

    def _build_params_precip(self, start: date, end: date):
        return {
            "sources": self.station_id,
            "elements": "sum(precipitation_amount P1D)",
            "referencetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T00:00:00Z",
            "timeresolutions": "P1D",
            "timeoffsets": "PT6H",
        }
 
    def get_range(self, start: date, end: date):
        """Get weather observations for a date range. Returns a dataframe"""
        data_path = self._data_path(start, end)
        if data_path.exists():
            return pd.read_parquet(data_path)
 
        hourly_data = self._get_response(self._build_params_hourly(start, end))
        precip_data = self._get_response(self._build_params_precip(start, end))
 
        df = self._convert_data((hourly_data or []) + (precip_data or []))

 
        if not df.empty:
            df.to_parquet(data_path, index=False)
 
        return df
 
    def get_range_chunked(self, start: date, end: date, chunk_days: int = 30):
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
 
        return pd.concat(dataframes, ignore_index=True).sort_values("valid_time").reset_index(drop=True)
 
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

 
client_id = os.getenv("FROST_CLIENT_ID")
client = FrostClient(client_id=client_id)



weather = client.get_range(date(2026, 3, 1), date(2026, 4, 1))
print(weather.head(20))
print(weather.describe())



def xd():
    strompris = StromprisClient()

    end = date.today()
    start = end - timedelta(days=1)
    df = strompris.get_range(start, end)
    print(df.head(50))
