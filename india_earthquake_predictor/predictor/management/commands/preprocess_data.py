from django.core.management.base import BaseCommand
import pandas as pd
import os

class Command(BaseCommand):
    help = 'Clean the downloaded earthquake data'

    def handle(self, *args, **options):
        try:
            df = pd.read_csv("assets/india_earthquakes.csv")
            df = df[['time', 'latitude', 'longitude', 'mag', 'depth']]
            df = df.dropna()
            os.makedirs('assets', exist_ok=True)
            df.to_csv("assets/cleaned_india_earthquakes.csv", index=False)
            self.stdout.write(self.style.SUCCESS("Data cleaned and saved as 'assets/cleaned_india_earthquakes.csv'"))
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR("Error: 'assets/india_earthquakes.csv' not found. Run download_data first."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error preprocessing data: {e}"))