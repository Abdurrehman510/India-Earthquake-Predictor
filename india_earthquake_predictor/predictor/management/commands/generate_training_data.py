from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
import os

class Command(BaseCommand):
    help = 'Generate training data with synthetic MMI labels'

    def add_arguments(self, parser):
        parser.add_argument('--max_earthquakes', type=int, default=1000)
        parser.add_argument('--points_per_eq', type=int, default=50)

    def handle(self, *args, **options):
        max_earthquakes = options['max_earthquakes']
        points_per_eq = options['points_per_eq']
        try:
            df = pd.read_csv("assets/cleaned_india_earthquakes.csv")
            self.stdout.write(f"Loaded {len(df)} earthquakes")
            
            if len(df) > max_earthquakes:
                df = df.sample(n=max_earthquakes, random_state=42)
                self.stdout.write(f"Sampled {max_earthquakes} earthquakes for processing")
            
            data = []
            for i, eq in enumerate(df.itertuples()):
                if i % 100 == 0:
                    self.stdout.write(f"Processing earthquake {i+1}/{len(df)}")
                lats, lons, distances, angles = self.generate_points(eq.latitude, eq.longitude, num_points=points_per_eq)
                if len(lats) == 0:
                    self.stdout.write(f"Skipping earthquake at index {i} due to point generation error")
                    continue
                for j in range(len(lats)):
                    mmi = self.simple_gmpe(eq.mag, distances[j])
                    mmi += np.random.normal(0, 0.5)
                    mmi = max(1, min(mmi, 12))
                    features = {
                        'magnitude': eq.mag + np.random.normal(0, 0.1),
                        'distance': distances[j],
                        'angle': angles[j],
                        'depth': eq.depth + np.random.normal(0, 2.0),
                        'latitude': lats[j],
                        'longitude': lons[j],
                        'label': 1 if mmi >= 6 else 0
                    }
                    data.append(features)
            
            train_df = pd.DataFrame(data)
            os.makedirs('assets', exist_ok=True)
            train_df.to_csv("assets/training_data.csv", index=False)
            self.stdout.write(self.style.SUCCESS(f"Training data generated with {len(train_df)} points and saved as 'assets/training_data.csv'"))
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR("Error: 'assets/cleaned_india_earthquakes.csv' not found. Run preprocess_data first."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error generating training data: {e}"))

    def simple_gmpe(self, magnitude, distance):
        mmi = 1.68 * magnitude - 3.29 - 0.0206 * distance
        return max(1, min(mmi, 12))

    def generate_points(self, epicenter_lat, epicenter_lon, num_points=50, max_distance=200):
        try:
            distances = np.random.uniform(0, max_distance, num_points)
            angles = np.random.uniform(0, 360, num_points)
            lat_offsets = distances * np.cos(np.radians(angles)) / 111
            lon_offsets = distances * np.sin(np.radians(angles)) / (111 * np.cos(np.radians(epicenter_lat)))
            lats = epicenter_lat + lat_offsets
            lons = epicenter_lon + lon_offsets
            return lats, lons, distances, angles
        except Exception as e:
            self.stdout.write(f"Error generating points for lat={epicenter_lat}, lon={epicenter_lon}: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])