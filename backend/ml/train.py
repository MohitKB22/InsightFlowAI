"""
ML Pipeline — Feature Engineering, Model Training, Evaluation, Serialization.
Uses XGBoost for fraud/anomaly detection on synthetic transaction data.
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    XGBOOST_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

CATEGORICAL_COLS = ["device_type", "event_type", "location"]
FEATURE_COLS: List[str] = []   # set after encoding

# ─────────────────────────────────────────────
# Synthetic dataset generator
# ─────────────────────────────────────────────

def generate_training_data(n_samples: int = 5000, fraud_rate: float = 0.08) -> pd.DataFrame:
    """
    Generate a labelled synthetic transaction dataset for training.
    Fraudulent transactions have distinctive statistical signatures.
    """
    np.random.seed(42)
    n_fraud  = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud

    # Normal transactions
    normal = pd.DataFrame({
        "transaction_amount": np.random.lognormal(mean=4.5, sigma=1.2, size=n_normal),
        "hour_of_day":        np.random.randint(6, 23, n_normal),
        "day_of_week":        np.random.randint(0, 7, n_normal),
        "is_weekend":         np.random.randint(0, 2, n_normal),
        "is_night":           np.random.choice([0, 1], n_normal, p=[0.85, 0.15]),
        "device_type":        np.random.choice(
                                  ["mobile", "desktop", "tablet", "api"],
                                  n_normal, p=[0.45, 0.35, 0.15, 0.05]
                              ),
        "event_type":         np.random.choice(
                                  ["purchase", "transfer", "withdrawal",
                                   "login", "failed_login"],
                                  n_normal, p=[0.5, 0.2, 0.1, 0.15, 0.05]
                              ),
        "location":           np.random.choice(
                                  ["New York", "Los Angeles", "Chicago",
                                   "Houston", "Dallas"],
                                  n_normal
                              ),
        "txn_count_1h":       np.random.poisson(lam=2, size=n_normal),
        "avg_amount_1h":      np.random.lognormal(mean=4.0, sigma=1.0, size=n_normal),
        "max_amount_1h":      np.random.lognormal(mean=4.5, sigma=1.2, size=n_normal),
        "failed_attempts_1h": np.random.poisson(lam=0.2, size=n_normal),
        "unique_locations_1h":np.random.randint(1, 3, n_normal),
        "label":              0,
    })

    # Fraudulent transactions — distinguishable patterns
    fraud = pd.DataFrame({
        "transaction_amount": np.random.lognormal(mean=8.5, sigma=1.5, size=n_fraud),
        "hour_of_day":        np.random.choice([1, 2, 3, 4], n_fraud),
        "day_of_week":        np.random.randint(0, 7, n_fraud),
        "is_weekend":         np.random.randint(0, 2, n_fraud),
        "is_night":           np.ones(n_fraud, dtype=int),
        "device_type":        np.random.choice(
                                  ["api", "mobile"],
                                  n_fraud, p=[0.75, 0.25]
                              ),
        "event_type":         np.random.choice(
                                  ["transfer", "withdrawal", "failed_login"],
                                  n_fraud, p=[0.50, 0.35, 0.15]
                              ),
        "location":           np.random.choice(
                                  ["Unknown", "Offshore", "VPN", "Lagos"],
                                  n_fraud
                              ),
        "txn_count_1h":       np.random.poisson(lam=15, size=n_fraud),
        "avg_amount_1h":      np.random.lognormal(mean=8.0, sigma=1.0, size=n_fraud),
        "max_amount_1h":      np.random.lognormal(mean=9.0, sigma=1.0, size=n_fraud),
        "failed_attempts_1h": np.random.poisson(lam=8, size=n_fraud),
        "unique_locations_1h":np.random.randint(3, 10, n_fraud),
        "label":              1,
    })

    df = pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=42)
    return df


# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────

class FeatureEngineer:
    """Transforms raw event data into ML-ready features."""

    def __init__(self) -> None:
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._encode_categoricals(df, fit=True)
        df = self._add_derived_features(df)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._encode_categoricals(df, fit=False)
        df = self._add_derived_features(df)
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                df[col] = "unknown"
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    df[col] = 0
                else:
                    known = set(le.classes_)
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in known else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "transaction_amount" in df.columns:
            df["log_amount"] = np.log1p(df["transaction_amount"])
            if "avg_amount_1h" in df.columns and df["avg_amount_1h"].sum() > 0:
                df["amount_vs_avg_ratio"] = df["transaction_amount"] / (
                    df["avg_amount_1h"].replace(0, 1)
                )
            else:
                df["amount_vs_avg_ratio"] = 1.0
        if "txn_count_1h" in df.columns and "failed_attempts_1h" in df.columns:
            df["fail_rate"] = df["failed_attempts_1h"] / (
                df["txn_count_1h"].replace(0, 1)
            )
        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        exclude = {"label", "event_id", "user_id", "session_id",
                   "timestamp", "ip_address", "merchant_id", "currency"}
        return [c for c in df.columns if c not in exclude]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "FeatureEngineer":
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────
# Model trainer
# ─────────────────────────────────────────────

class FraudDetectionTrainer:
    """
    End-to-end training pipeline:
    data → feature engineering → XGBoost → evaluation → serialization.
    """

    def __init__(self) -> None:
        self.feature_engineer = FeatureEngineer()
        self.scaler           = StandardScaler()
        self.model: Optional[Any] = None
        self.feature_names: List[str] = []
        self.metrics: Dict[str, Any] = {}

    def _build_model(self) -> Any:
        if XGBOOST_AVAILABLE:
            return XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=11,   # roughly 1/fraud_rate
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        # Fallback to sklearn GradientBoosting
        from sklearn.ensemble import GradientBoostingClassifier
        logger.warning("XGBoost not available — falling back to GradientBoostingClassifier")
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )

    def train(self, df: Optional[pd.DataFrame] = None, n_samples: int = 5000) -> Dict[str, Any]:
        """
        Train the fraud detection model.

        Args:
            df: Pre-built DataFrame. If None, synthetic data is generated.
            n_samples: Number of samples for synthetic generation.

        Returns:
            dict of evaluation metrics.
        """
        if df is None:
            logger.info("Generating %d synthetic training samples…", n_samples)
            df = generate_training_data(n_samples=n_samples)

        logger.info("Dataset shape: %s | Fraud rate: %.2f%%",
                    df.shape, df["label"].mean() * 100)

        # Feature engineering
        df_feat = self.feature_engineer.fit_transform(df)
        self.feature_names = self.feature_engineer.get_feature_names(df_feat)

        X = df_feat[self.feature_names].values
        y = df_feat["label"].values

        # Scale
        X = self.scaler.fit_transform(X)

        # Train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info("Training XGBoost on %d samples…", len(X_train))
        self.model = self._build_model()
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred      = self.model.predict(X_test)
        y_prob      = self.model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
            "f1":        round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
            "train_size": len(X_train),
            "test_size":  len(X_test),
            "fraud_rate": round(float(y.mean()), 4),
        }

        logger.info("── Evaluation Metrics ──────────────────")
        for k, v in self.metrics.items():
            logger.info("  %-15s : %s", k, v)
        logger.info("────────────────────────────────────────")
        logger.info("\n%s", classification_report(y_test, y_pred,
                                                  target_names=["Normal", "Fraud"]))
        logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        self.metrics["cv_roc_auc_mean"] = round(float(cv_scores.mean()), 4)
        self.metrics["cv_roc_auc_std"]  = round(float(cv_scores.std()), 4)
        logger.info("CV ROC-AUC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

        return self.metrics

    def save_artifacts(self) -> None:
        """Serialize model, scaler, and feature names to disk."""
        paths = {
            "model":         MODELS_DIR / "fraud_model.pkl",
            "scaler":        MODELS_DIR / "scaler.pkl",
            "feature_names": MODELS_DIR / "feature_names.pkl",
            "feature_eng":   MODELS_DIR / "feature_engineer.pkl",
        }
        with open(paths["model"], "wb") as f:
            pickle.dump(self.model, f)
        with open(paths["scaler"], "wb") as f:
            pickle.dump(self.scaler, f)
        with open(paths["feature_names"], "wb") as f:
            pickle.dump(self.feature_names, f)
        self.feature_engineer.save(paths["feature_eng"])
        logger.info("Artifacts saved to: %s", MODELS_DIR)

    def load_artifacts(self) -> None:
        with open(MODELS_DIR / "fraud_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(MODELS_DIR / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(MODELS_DIR / "feature_names.pkl", "rb") as f:
            self.feature_names = pickle.load(f)
        self.feature_engineer = FeatureEngineer.load(MODELS_DIR / "feature_engineer.pkl")
        logger.info("Artifacts loaded from: %s", MODELS_DIR)


# ─────────────────────────────────────────────
# Inference engine (used in streaming layer)
# ─────────────────────────────────────────────

class InferenceEngine:
    """
    Lightweight wrapper for real-time inference.
    Loaded once at startup; predict() called per event.
    """

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.feature_engineer: Optional[FeatureEngineer] = None
        self._loaded = False

    def load(self) -> None:
        trainer = FraudDetectionTrainer()
        trainer.load_artifacts()
        self.model            = trainer.model
        self.scaler           = trainer.scaler
        self.feature_names    = trainer.feature_names
        self.feature_engineer = trainer.feature_engineer
        self._loaded = True
        logger.info("InferenceEngine ready.")

    def predict(self, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud probability for a single event.

        Args:
            event_dict: Raw event as dict (keys matching RawEvent schema).

        Returns:
            dict with fraud_probability, is_fraud, confidence.
        """
        if not self._loaded:
            self.load()

        df = pd.DataFrame([event_dict])
        df_feat = self.feature_engineer.transform(df)

        # Align columns
        missing = [c for c in self.feature_names if c not in df_feat.columns]
        for m in missing:
            df_feat[m] = 0
        df_feat = df_feat[self.feature_names]

        X = self.scaler.transform(df_feat.values)
        prob = float(self.model.predict_proba(X)[0, 1])
        return {
            "fraud_probability": round(prob, 4),
            "is_fraud":          prob >= 0.5,
            "confidence":        round(max(prob, 1 - prob), 4),
        }

    def predict_batch(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict(e) for e in events]


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def train_and_save(n_samples: int = 5000) -> Dict[str, Any]:
    """Train the model and save artifacts. Returns metrics."""
    trainer = FraudDetectionTrainer()
    metrics = trainer.train(n_samples=n_samples)
    trainer.save_artifacts()
    return metrics


if __name__ == "__main__":
    metrics = train_and_save(n_samples=5000)
    print("\n=== Training Complete ===")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")
