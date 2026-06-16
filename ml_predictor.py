import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import random

class ParkingPredictor:
    """
    ML-based parking availability predictor.
    Uses Random Forest trained on synthetic historical occupancy data.
    In production: replace synthetic data with real logged occupancy.
    """

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self._train()

    def _generate_training_data(self):
        """Generate realistic synthetic occupancy data."""
        X, y = [], []
        for _ in range(2000):
            hour    = random.randint(0, 23)
            day     = random.randint(0, 6)    # 0=Mon, 6=Sun
            month   = random.randint(1, 12)
            is_weekend = 1 if day >= 5 else 0

            # Realistic occupancy model
            if 0 <= hour < 7:
                base = 5
            elif 7 <= hour < 9:
                base = 55 + (hour - 7) * 20   # morning rush
            elif 9 <= hour < 12:
                base = 75
            elif 12 <= hour < 14:
                base = 85                       # lunch peak
            elif 14 <= hour < 17:
                base = 70
            elif 17 <= hour < 19:
                base = 80                       # evening rush
            elif 19 <= hour < 22:
                base = 50
            else:
                base = 15

            if is_weekend:
                base = base * 0.6

            noise = random.gauss(0, 8)
            occupancy = max(0, min(100, base + noise))
            X.append([hour, day, month, is_weekend])
            y.append(occupancy)
        return np.array(X), np.array(y)

    def _train(self):
        X, y = self._generate_training_data()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_availability(self, hour: int, day: int, month: int = None):
        if month is None:
            import datetime
            month = datetime.datetime.now().month
        is_weekend = 1 if day >= 5 else 0
        features = np.array([[hour, day, month, is_weekend]])
        features_scaled = self.scaler.transform(features)
        occupied_pct = float(self.model.predict(features_scaled)[0])
        occupied_pct = max(0, min(100, occupied_pct))
        available_pct = 100 - occupied_pct

        # Map to spot counts (20 total)
        total = 20
        occupied_count  = round(total * occupied_pct / 100)
        available_count = total - occupied_count

        confidence = 0.91 - abs(hour - 12) * 0.01  # slightly less confident off-peak

        return {
            "occupied_pct":   round(occupied_pct, 1),
            "available_pct":  round(available_pct, 1),
            "occupied_spots":  occupied_count,
            "available_spots": available_count,
            "total_spots":     total,
            "confidence":      round(max(0.7, confidence), 2),
            "hour":  hour,
            "day":   day,
        }

    def get_best_times(self, day: int):
        """Return best hours to park (lowest predicted occupancy)."""
        predictions = []
        for h in range(7, 23):
            pred = self.predict_availability(h, day)
            predictions.append((h, pred["available_pct"]))
        predictions.sort(key=lambda x: -x[1])
        return [{"hour": h, "label": f"{h:02d}:00", "available_pct": round(a, 1)}
                for h, a in predictions[:5]]
