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
        """Get hourly strom prices for one day. Returns a dataframe"""
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
        """Get hourly strom prices from a range of days"""
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
    

class NveClient:
    """Fetches dam water levels from NVE"""
 
    BASE_URL = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOffentligData"
 
    def __init__(
        self,
        retry_limit: int = 3,
        backoff_factor: float = 0.5,
        timeout: float = 30,
        omr_type: str = "EL", 
        omrnr: int = 1,
        ):
        retry = Retry(total=retry_limit, backoff_factor=backoff_factor)
        transport = RetryTransport(retry=retry)
        self.client = httpx.Client(transport=transport, timeout=timeout)
        self.omr_type = omr_type
        self.omrnr = omrnr
 
    def get_all(self):
        """Get full weekly level history for an area. Returns a dataframe"""
        
        data = self._get_response()
        df = self._convert_data(data)
        
        return df
 
    def _get_response(self):
        """HTTP response"""
        try:
            res = self.client.get(self.BASE_URL)
            res.raise_for_status()
            time.sleep(1)
            return res.json()
        except httpx.HTTPError as exc:
            print(f"HTTP exception: {exc}")
            return None
 
    def _convert_data(self, data: list[dict]):
        """Convert data from json to dataframe, filtered to the chosen area"""
        if not data:
            return pd.DataFrame()
 
        df = pd.DataFrame(data)
        df = df[(df["omrType"] == self.omr_type) & (df["omrnr"] == self.omrnr)]
 
        if df.empty:
            return df
 
        df["valid_time"] = pd.to_datetime(df["dato_Id"], utc=True)
 
        return df[["valid_time", "fyllingsgrad", "fylling_TWh", "kapasitet_TWh"]].sort_values("valid_time").reset_index(drop=True)
 


def merge_data(strompriser: pd.DataFrame, weather: pd.DataFrame, magasin: pd.DataFrame) -> pd.DataFrame:
    """Combine electricity prices, weather data, water dam data to one format: timestamp, prisområde, pris_nok_kwh, temperatur_c, nedbor_mm_24h, vind_m_s, fylling_TWh, kapasitet_TWh"""
    df = strompriser.merge(
        weather,
        left_on="timestamp",
        right_on="valid_time",
        how="left",
        )
    
    df = df.drop(columns=["valid_time", "pris_eur_kwh"])
    
    df = df.merge(
        magasin,
        left_on="timestamp",
        right_on="valid_time",
        how="left",
        )
    
    df = df.drop(columns=["valid_time", "fylling_TWh", "kapasitet_TWh"])
    
    df["nedbor_mm_24h"] = df["nedbor_mm_24h"].ffill()
    df["fyllingsgrad"] = df["fyllingsgrad"].ffill()
 
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_data_range(area: str = "NO1", start: date = date(2025, 6, 1), end: date | None = None, data_dir: str | Path = "data"):
    """Fetch strompriser, weather data, magasin data for a time range, merge into one dataframe and save as .parquet"""
    
    if end is None:
        end = date.today() - timedelta(days=1)
        
        
    strompris = StromprisClient()
    
    client_id = os.getenv("FROST_CLIENT_ID")
    if not client_id:
        raise ValueError("FROST_CLIENT_ID er ikke satt — sjekk at .env lastes inn før dette kjøres")


    frost = FrostClient(client_id=client_id)
    nve = NveClient()
    
    print(f"Fetching data for time range: {start} -> {end}")
    print(f"Fetching strompriser...")
    strompriser = strompris.get_range(start, end)
    print(strompriser.tail(10))
    
    print(f"Fetching weather data...")
    weather = frost.get_range_chunked(start, end, chunk_days=90)
    print(weather.tail(10))
    
    print(f"Fetching magasin data...")
    magasin = nve.get_all()
    print(magasin.tail(10))
    
    if strompriser.empty:
        print("No strompriser, exiting...")
        return None
    
    merged = merge_data(strompriser=strompriser, weather=weather, magasin=magasin)
    
    output_dir = Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{area}_historical.parquet"
    
    merged.to_parquet(output_path, index=False)
    print(f"Saved {len(merged)} rows for time range: ( {start} to {end} ) -> {output_path}")
    
    return merged


# todo: find corresponding weather stations to strompris areas

if __name__ == "__main__":
    start = date(2022, 1, 1)
    end = date.today() - timedelta(days=1)

    data = fetch_data_range(start=start, end=end)
    print(data.tail(10))