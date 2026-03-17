from django.core.management.base import BaseCommand
import requests
import pandas as pd
import os
from io import StringIO

class Command(BaseCommand):
    help = 'Download earthquake data for India from USGS'

    def handle(self, *args, **options):
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        base_params = {
            "format": "csv",
            "minlatitude": 8,
            "maxlatitude": 37,
            "minlongitude": 68,
            "maxlongitude": 97,
            "minmagnitude": 3.0
        }
        years = list(range(1900, 2024, 10))
        all_data = []
        
        for start_year in years:
            end_year = min(start_year + 9, 2023)
            params = base_params.copy()
            params["starttime"] = f"{start_year}-01-01"
            params["endtime"] = f"{end_year}-12-31"
            try:
                self.stdout.write(f"Downloading data for {start_year}–{end_year}")
                response = requests.get(url, params=params)
                response.raise_for_status()
                if response.text.strip():
                    df_chunk = pd.read_csv(StringIO(response.text))
                    if not df_chunk.empty:
                        all_data.append(df_chunk)
                    else:
                        self.stdout.write(f"No data found for {start_year}–{end_year}")
                else:
                    self.stdout.write(f"Empty response for {start_year}–{end_year}")
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error downloading data for {start_year}–{end_year}: {e}"))
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            os.makedirs('assets', exist_ok=True)
            df.to_csv("assets/india_earthquakes.csv", index=False)
            self.stdout.write(self.style.SUCCESS(f"Data downloaded and saved as 'assets/india_earthquakes.csv' with {len(df)} events"))
        else:
            if os.path.exists("assets/india_earthquakes.csv"):
                self.stdout.write("No new data downloaded. Using existing 'assets/india_earthquakes.csv'.")
            else:
                raise Exception("No data downloaded and no existing 'assets/india_earthquakes.csv' found.")