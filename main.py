import httpx, pandas as pd, time
from datetime import date, timedelta
from pathlib import Path



class StromprisClient:
    """
    Fetches prices for electricity from hvakosterstrommen.no
    """
    BASE_URL = "https://www.hvakosterstrommen.no/api/v1/prices"
    AREAS = {"NO1", "NO2", "NO3", "NO4", "NO5"}
    
    def __init__(self, area: str = "NO1", data_dir: Path = "data/strompris"):      
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
            print(f"{current} done.")
            current += timedelta(days=1)
            
            
        if not dataframes:
            return pd.DataFrame()
        
        return pd.concat(dataframes, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        
    def _get_response(self, url):
        """HTTP response"""
        res = httpx.get(url)
        time.sleep(0.5)
        return res.json()
    
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
    
strompris = StromprisClient()

end = date.today()
start = end - timedelta(days=365)
history = strompris.get_range(start, end)
print(history.describe())


