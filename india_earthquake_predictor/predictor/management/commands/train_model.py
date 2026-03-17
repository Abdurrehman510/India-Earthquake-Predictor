from django.core.management.base import BaseCommand
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

class Command(BaseCommand):
    help = 'Train and save the Random Forest model'

    def handle(self, *args, **options):
        try:
            df = pd.read_csv("assets/training_data.csv")
            self.stdout.write(f"Loaded training data with {len(df)} samples")
            X = df[['magnitude', 'distance', 'angle', 'depth']]
            y = df['label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.stdout.write(f"Model Accuracy: {accuracy:.2f}")
            os.makedirs('assets', exist_ok=True)
            joblib.dump(model, "assets/earthquake_model.pkl")
            self.stdout.write(self.style.SUCCESS("Model saved as 'assets/earthquake_model.pkl'"))
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR("Error: 'assets/training_data.csv' not found. Run generate_training_data first."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error training model: {e}"))