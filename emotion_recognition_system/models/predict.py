"""
Emotion prediction from raw physiological sensor data.

This is the core inference module — it takes raw EDA (and optionally PPG)
sensor readings and returns an emotion prediction (excitement, neutral or frustration)
with confidence scores.

Usage:
    from models.predict import EmotionPredictor

    # Train on existing data
    predictor = EmotionPredictor()
    predictor.train_from_features("features/emotion_features.csv")

    # Predict from raw sensor window (30 seconds at 64 Hz = 1920 samples)
    result = predictor.predict_from_raw(eda_window=eda_array)
    print(result["prediction"])   # "excitement", "neutral" or "frustration"
    print(result["confidence"])   # e.g. 0.73
    print(result["probabilities"])  # {"excitement": 0.73, "frustration": 0.18, "neutral": 0.09}

    # Save/load for deployment
    predictor.save("models/trained_model.pkl")
    predictor = EmotionPredictor.load("models/trained_model.pkl")
"""

import pickle
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestClassifier

from models.config import RF_PARAMS, EDA_FEATURES, SAMPLE_RATE


class EmotionPredictor:
    """End-to-end emotion predictor: raw sensor data in, emotion label out."""

    def __init__(self, sample_rate=SAMPLE_RATE, window_sec=30):
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.window_samples = int(sample_rate * window_sec)
        self.model = None
        self.feature_medians = None
        self.feature_cols = list(EDA_FEATURES)

    

    def train_from_features(self, csv_path):
        """Train the model from pre-extracted windowed feature CSV."""
        df = pd.read_csv(csv_path)

        X = df[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        y = df["label"]

        # Store medians for imputation at inference time
        self.feature_medians = X.median()
        X = X.fillna(self.feature_medians)

        self.model = RandomForestClassifier(**RF_PARAMS)
        self.model.fit(X, y)

        acc = self.model.score(X, y)
        print(f"Trained on {len(X)} windows, training accuracy: {acc:.3f}")
        print(f"Classes: {list(self.model.classes_)}")
        return self

    

    def _extract_eda_features(self, eda_raw):
        """Extract EDA features from a raw sensor window."""
        eda = np.asarray(eda_raw, dtype=np.float64)
        eda = eda[np.isfinite(eda)]

        if eda.size < max(8, int(self.sample_rate * 3)):
            return {k: np.nan for k in EDA_FEATURES}

        nyq = 0.5 * self.sample_rate

        # Low-pass at 5 Hz to remove sensor noise
        b, a = butter(4, 5.0 / nyq, btype="low")
        cleaned = filtfilt(b, a, eda)

        # Tonic/phasic decomposition
        cutoff = min(0.05 / nyq, 0.999)
        b2, a2 = butter(4, cutoff, btype="low")
        tonic = filtfilt(b2, a2, cleaned)
        phasic = cleaned - tonic

        # Tonic slope
        t = np.arange(len(tonic), dtype=np.float64) / self.sample_rate
        slope = float(np.polyfit(t, tonic, 1)[0]) if tonic.size > 1 else 0.0

        # SCR peak detection from phasic component
        phasic_std = float(np.std(phasic))
        prom = max(0.05 * phasic_std, 1e-6)
        peaks, props = find_peaks(
            phasic, prominence=prom,
            distance=max(1, int(0.5 * self.sample_rate)),
        )
        peak_prom = props.get("prominences", np.array([], dtype=np.float64))

        # First derivative
        deriv = np.diff(cleaned)

        return {
            "eda_mean_scl": float(np.mean(tonic)),
            "eda_slope_scl": slope,
            "eda_num_scr": int(len(peaks)),
            "eda_mean_scr_amp": float(np.mean(peak_prom)) if peak_prom.size else np.nan,
            "eda_sum_scr_amp": float(np.sum(peak_prom)) if peak_prom.size else 0.0,
            "eda_variance": float(np.var(cleaned)),
            "eda_range": float(np.max(cleaned) - np.min(cleaned)),
            "eda_deriv_mean": float(np.mean(deriv)) if deriv.size else 0.0,
        }

    

    def predict_from_raw(self, eda_window):
        """
        Predict emotion from a raw EDA sensor window.

        Args:
            eda_window: array-like of raw EDA values (ideally 30s × 64Hz = 1920 samples)

        Returns:
            dict with keys:
                prediction: "excitement", "neutral" or "frustration"
                confidence: float (probability of predicted class)
                probabilities: dict mapping class names to probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_from_features() first.")

        features = self._extract_eda_features(eda_window)
        feat_df = pd.DataFrame([features])[self.feature_cols]
        feat_df = feat_df.fillna(self.feature_medians)

        prediction = self.model.predict(feat_df)[0]
        probas = self.model.predict_proba(feat_df)[0]
        proba_dict = dict(zip(self.model.classes_, probas.round(4)))

        return {
            "prediction": prediction,
            "confidence": float(max(probas)),
            "probabilities": proba_dict,
        }

    def predict_stream(self, eda_signal, stride_sec=15):
        """
        Predict emotions across a full recording using a sliding window.

        Args:
            eda_signal: full raw EDA array
            stride_sec: seconds between window starts

        Returns:
            list of dicts, each with prediction, confidence, time_start, time_end
        """
        eda = np.asarray(eda_signal, dtype=np.float64)
        stride = int(stride_sec * self.sample_rate)
        results = []
        start = 0

        while start + self.window_samples <= len(eda):
            window = eda[start:start + self.window_samples]
            result = self.predict_from_raw(window)
            result["time_start_sec"] = start / self.sample_rate
            result["time_end_sec"] = (start + self.window_samples) / self.sample_rate
            results.append(result)
            start += stride

        return results

    

    def save(self, filepath):
        """Save trained model and configuration to disk."""
        state = {
            "model": self.model,
            "feature_medians": self.feature_medians,
            "feature_cols": self.feature_cols,
            "sample_rate": self.sample_rate,
            "window_sec": self.window_sec,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load a trained predictor from disk."""
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        predictor = cls(
            sample_rate=state["sample_rate"],
            window_sec=state["window_sec"],
        )
        predictor.model = state["model"]
        predictor.feature_medians = state["feature_medians"]
        predictor.feature_cols = state["feature_cols"]
        return predictor
