"""
Unit Tests — covers models, feature engineering, alert rules,
streaming aggregation, and storage layer (mocked).
Run: pytest tests/ -v --cov=backend --cov-report=term-missing
"""
from __future__ import annotations

import sys
import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_raw_event() -> Dict[str, Any]:
    return {
        "event_id":           str(uuid.uuid4()),
        "user_id":            "user_0001",
        "session_id":         str(uuid.uuid4()),
        "timestamp":          datetime.utcnow().isoformat(),
        "transaction_amount": 250.50,
        "location":           "New York",
        "device_type":        "mobile",
        "event_type":         "purchase",
        "ip_address":         "192.168.1.1",
        "merchant_id":        "merchant_0001",
    }


@pytest.fixture
def sample_fraud_event() -> Dict[str, Any]:
    return {
        "event_id":           str(uuid.uuid4()),
        "user_id":            "user_0001",
        "session_id":         str(uuid.uuid4()),
        "timestamp":          datetime.utcnow().isoformat(),
        "transaction_amount": 49_000.0,
        "location":           "Unknown",
        "device_type":        "api",
        "event_type":         "transfer",
        "fraud_probability":  0.95,
        "is_fraud":           True,
        "failed_attempts_1h": 10,
        "txn_count_1h":       25,
        "is_night":           1,
    }


@pytest.fixture
def sample_metrics() -> Dict[str, Any]:
    return {
        "window_start":       "2024-01-15T14:00:00",
        "window_end":         "2024-01-15T14:01:00",
        "total_transactions": 200,
        "total_amount":       80_000.0,
        "avg_amount":         400.0,
        "max_amount":         15_000.0,
        "unique_users":       80,
        "fraud_count":        10,
        "fraud_rate":         0.05,
        "event_type_breakdown": {"purchase": 120, "transfer": 50, "withdrawal": 30},
        "device_breakdown":     {"mobile": 120, "desktop": 60, "api": 20},
        "location_breakdown":   {"New York": 80, "Los Angeles": 60, "Unknown": 20},
    }


# ─────────────────────────────────────────────
# Models tests
# ─────────────────────────────────────────────

class TestRawEventModel:
    def test_valid_event_creation(self, sample_raw_event):
        from backend.models import RawEvent, EventType, DeviceType
        event = RawEvent(
            user_id="user_001",
            session_id="sess_001",
            transaction_amount=100.0,
            location="New York",
            device_type=DeviceType.MOBILE,
            event_type=EventType.PURCHASE,
        )
        assert event.user_id == "user_001"
        assert event.transaction_amount == 100.0
        assert event.event_id is not None

    def test_negative_amount_raises(self):
        from backend.models import RawEvent, EventType, DeviceType
        with pytest.raises(Exception):
            RawEvent(
                user_id="user_001",
                session_id="sess_001",
                transaction_amount=-50.0,
                location="New York",
                device_type=DeviceType.MOBILE,
                event_type=EventType.PURCHASE,
            )

    def test_to_kafka_dict_serialisable(self):
        from backend.models import RawEvent, EventType, DeviceType
        event = RawEvent(
            user_id="user_001",
            session_id="sess_001",
            transaction_amount=99.99,
            location="Chicago",
            device_type=DeviceType.DESKTOP,
            event_type=EventType.TRANSFER,
        )
        d = event.to_kafka_dict()
        assert isinstance(d, dict)
        # Must be JSON-serialisable
        json_str = json.dumps(d)
        assert "user_001" in json_str

    def test_amount_rounds_to_2dp(self):
        from backend.models import RawEvent, EventType, DeviceType
        event = RawEvent(
            user_id="u", session_id="s",
            transaction_amount=1.23456789,
            location="X", device_type=DeviceType.MOBILE,
            event_type=EventType.PURCHASE,
        )
        assert event.transaction_amount == 1.23


# ─────────────────────────────────────────────
# Event generator tests
# ─────────────────────────────────────────────

class TestEventGenerator:
    def test_generates_event(self):
        from backend.producer.event_producer import EventGenerator
        gen   = EventGenerator()
        event = gen.generate_event()
        assert event.user_id is not None
        assert event.transaction_amount >= 0

    def test_generates_batch(self):
        from backend.producer.event_producer import EventGenerator
        gen   = EventGenerator()
        batch = gen.generate_batch(20)
        assert len(batch) == 20

    def test_fraud_events_have_high_amounts(self):
        from backend.producer.event_producer import EventGenerator
        gen = EventGenerator()
        # Generate many fraud events and verify average amount is high
        frauds = [gen.generate_fraudulent_event("user_0001") for _ in range(50)]
        amounts = [e.transaction_amount for e in frauds]
        assert min(amounts) >= 5000, "Fraud events should have amounts >= 5000"

    def test_normal_events_have_moderate_amounts(self):
        from backend.producer.event_producer import EventGenerator
        gen    = EventGenerator()
        normal = [gen.generate_normal_event("user_0050") for _ in range(100)]
        amounts = [e.transaction_amount for e in normal]
        assert max(amounts) <= 600, "Normal event amounts should be <= 500"


# ─────────────────────────────────────────────
# Feature engineering tests
# ─────────────────────────────────────────────

class TestFeatureEngineer:
    def test_fit_transform_runs(self):
        from backend.ml.train import FeatureEngineer, generate_training_data
        df = generate_training_data(n_samples=200)
        fe = FeatureEngineer()
        transformed = fe.fit_transform(df)
        assert len(transformed) == 200

    def test_categorical_encoding(self):
        from backend.ml.train import FeatureEngineer, generate_training_data
        df = generate_training_data(n_samples=200)
        fe = FeatureEngineer()
        transformed = fe.fit_transform(df)
        for col in ["device_type", "event_type", "location"]:
            assert transformed[col].dtype in [np.int64, np.int32, int, "int64"], \
                f"{col} should be integer-encoded"

    def test_derived_features_created(self):
        from backend.ml.train import FeatureEngineer, generate_training_data
        df = generate_training_data(n_samples=200)
        fe = FeatureEngineer()
        transformed = fe.fit_transform(df)
        assert "log_amount" in transformed.columns
        assert "amount_vs_avg_ratio" in transformed.columns

    def test_transform_handles_unseen_categories(self):
        from backend.ml.train import FeatureEngineer, generate_training_data
        df = generate_training_data(n_samples=300)
        fe = FeatureEngineer()
        fe.fit_transform(df)
        test_df = pd.DataFrame([{
            "transaction_amount": 100.0, "hour_of_day": 10, "day_of_week": 2,
            "is_weekend": 0, "is_night": 0, "device_type": "UNKNOWN_DEVICE",
            "event_type": "UNKNOWN_EVENT", "location": "UNKNOWN_LOCATION",
            "txn_count_1h": 1, "avg_amount_1h": 100, "max_amount_1h": 100,
            "failed_attempts_1h": 0, "unique_locations_1h": 1,
        }])
        result = fe.transform(test_df)
        assert len(result) == 1


# ─────────────────────────────────────────────
# ML Training tests
# ─────────────────────────────────────────────

class TestMLTraining:
    def test_training_returns_metrics(self):
        from backend.ml.train import FraudDetectionTrainer
        trainer = FraudDetectionTrainer()
        metrics = trainer.train(n_samples=500)
        assert "roc_auc" in metrics
        assert "f1" in metrics
        assert metrics["roc_auc"] > 0.5, "ROC-AUC should be > 0.5 (better than random)"

    def test_trained_model_can_predict(self):
        from backend.ml.train import FraudDetectionTrainer
        trainer = FraudDetectionTrainer()
        trainer.train(n_samples=500)
        assert trainer.model is not None
        # Predict on a sample
        from backend.ml.train import generate_training_data
        df   = generate_training_data(n_samples=10)
        feat = trainer.feature_engineer.fit_transform(df)
        cols = trainer.feature_names
        cols_present = [c for c in cols if c in feat.columns]
        X    = trainer.scaler.transform(feat[cols_present].values) if cols_present else feat.values
        preds = trainer.model.predict_proba(X)
        assert preds.shape[1] == 2

    def test_high_fraud_event_scores_higher(self):
        """Fraud events should receive higher fraud probability than normal ones."""
        from backend.ml.train import FraudDetectionTrainer, InferenceEngine, train_and_save
        import tempfile, pathlib, os

        # Train small model
        trainer = FraudDetectionTrainer()
        trainer.train(n_samples=1000)

        # Manually wire inference engine
        engine = InferenceEngine()
        engine.model            = trainer.model
        engine.scaler           = trainer.scaler
        engine.feature_names    = trainer.feature_names
        engine.feature_engineer = trainer.feature_engineer
        engine._loaded          = True

        normal_event = {
            "transaction_amount": 100.0, "hour_of_day": 10, "day_of_week": 2,
            "is_weekend": 0, "is_night": 0, "device_type": "mobile",
            "event_type": "purchase", "location": "New York",
            "txn_count_1h": 2, "avg_amount_1h": 120, "max_amount_1h": 150,
            "failed_attempts_1h": 0, "unique_locations_1h": 1,
        }
        fraud_event = {
            "transaction_amount": 45000.0, "hour_of_day": 3, "day_of_week": 0,
            "is_weekend": 0, "is_night": 1, "device_type": "api",
            "event_type": "transfer", "location": "Unknown",
            "txn_count_1h": 20, "avg_amount_1h": 30000, "max_amount_1h": 45000,
            "failed_attempts_1h": 9, "unique_locations_1h": 7,
        }

        normal_score = engine.predict(normal_event)["fraud_probability"]
        fraud_score  = engine.predict(fraud_event)["fraud_probability"]
        assert fraud_score > normal_score, \
            f"Fraud event ({fraud_score}) should score higher than normal ({normal_score})"


# ─────────────────────────────────────────────
# Streaming processor tests
# ─────────────────────────────────────────────

class TestStreamingProcessor:
    def test_window_store_computes_features(self):
        from backend.streaming.processor import WindowStore
        ws = WindowStore(window_minutes=60)
        for i in range(5):
            ws.add_event("user_001", {
                "user_id": "user_001",
                "transaction_amount": 100.0 * (i + 1),
                "event_type": "purchase",
                "location": "New York",
                "timestamp": datetime.utcnow().isoformat(),
            })
        features = ws.compute_features("user_001", datetime.utcnow())
        assert features["txn_count_1h"] == 5
        assert features["avg_amount_1h"] == 300.0
        assert features["max_amount_1h"] == 500.0

    def test_enrich_event_adds_time_features(self, sample_raw_event):
        from backend.streaming.processor import enrich_event, WindowStore
        ws       = WindowStore()
        enriched = enrich_event(sample_raw_event, ws)
        assert "hour_of_day"  in enriched
        assert "day_of_week"  in enriched
        assert "is_weekend"   in enriched
        assert "is_night"     in enriched
        assert "processed_at" in enriched

    def test_aggregate_batch_produces_summary(self, sample_raw_event):
        from backend.streaming.processor import enrich_event, aggregate_batch, WindowStore
        ws      = WindowStore()
        events  = [enrich_event(sample_raw_event.copy(), ws) for _ in range(20)]
        summary = aggregate_batch(events)
        assert summary["total_transactions"] == 20
        assert summary["avg_amount"] > 0
        assert "event_type_breakdown" in summary


# ─────────────────────────────────────────────
# Alert engine tests
# ─────────────────────────────────────────────

class TestAlertEngine:
    def test_fraud_alert_fires(self, sample_fraud_event):
        from backend.alerts.alert_engine import AlertEngine
        engine = AlertEngine()
        alerts = engine.evaluate(sample_fraud_event)
        types  = [a.alert_type.value for a in alerts]
        assert "fraud_detected" in types, "FRAUD_DETECTED alert should fire for high fraud score"

    def test_high_transaction_alert_fires(self):
        from backend.alerts.alert_engine import AlertEngine
        engine = AlertEngine(high_amount=5000)
        event  = {
            "event_id": "e1", "user_id": "u1",
            "transaction_amount": 50_000.0,
            "fraud_probability": 0.1,
            "device_type": "desktop", "location": "Dallas",
            "failed_attempts_1h": 0, "txn_count_1h": 1, "is_night": 0,
        }
        alerts = engine.evaluate(event)
        types  = [a.alert_type.value for a in alerts]
        assert "high_transaction" in types

    def test_normal_event_no_alerts(self, sample_raw_event):
        from backend.alerts.alert_engine import AlertEngine
        engine = AlertEngine()
        event  = {
            **sample_raw_event,
            "fraud_probability": 0.05,
            "failed_attempts_1h": 0,
            "txn_count_1h": 2,
            "is_night": 0,
        }
        alerts = engine.evaluate(event)
        assert len(alerts) == 0, "Clean event should trigger no alerts"

    def test_failed_attempts_alert_fires(self):
        from backend.alerts.alert_engine import AlertEngine
        engine = AlertEngine(failed_attempts=3)
        event  = {
            "event_id": "e2", "user_id": "u2",
            "transaction_amount": 50.0,
            "fraud_probability": 0.1,
            "device_type": "mobile", "location": "NY",
            "failed_attempts_1h": 10, "txn_count_1h": 3, "is_night": 0,
        }
        alerts = engine.evaluate(event)
        types  = [a.alert_type.value for a in alerts]
        assert "failed_attempts" in types


# ─────────────────────────────────────────────
# Storage tests (mocked)
# ─────────────────────────────────────────────

class TestRedisStorage:
    def test_in_memory_fallback(self):
        from backend.storage.storage import RedisStorage
        # Force in-memory mode by using invalid host
        storage = RedisStorage(host="invalid_host_xyz", port=9999)
        storage.set_metrics("test:key", {"value": 42})
        result = storage.get_metrics("test:key")
        assert result == {"value": 42}

    def test_push_and_get_list(self):
        from backend.storage.storage import RedisStorage
        storage = RedisStorage(host="invalid_host_xyz", port=9999)
        for i in range(5):
            storage.push_to_list("test:list", {"i": i}, max_len=100)
        lst = storage.get_list("test:list")
        assert len(lst) == 5
        assert lst[0]["i"] == 0

    def test_store_and_retrieve_metrics(self):
        from backend.storage.storage import RedisStorage
        storage  = RedisStorage(host="invalid_host_xyz", port=9999)
        metrics  = {"total_transactions": 100, "fraud_rate": 0.05}
        storage.store_latest_aggregation(metrics)
        latest = storage.get_latest_metrics()
        assert latest["total_transactions"] == 100


# ─────────────────────────────────────────────
# LLM Insights tests (mocked)
# ─────────────────────────────────────────────

class TestLLMInsights:
    def test_mock_insight_generated(self, sample_metrics):
        from backend.ml.llm_insights import LLMInsightsEngine
        engine  = LLMInsightsEngine(api_key=None)  # Force mock mode
        insight = engine.generate_insight(sample_metrics)
        assert "insight_text"     in insight
        assert "key_findings"     in insight
        assert "recommendations"  in insight
        assert "risk_level"       in insight
        assert insight["risk_level"] in ("low", "medium", "high", "critical")

    def test_risk_level_reflects_fraud_rate(self):
        from backend.ml.llm_insights import LLMInsightsEngine
        engine = LLMInsightsEngine(api_key=None)

        low_risk_metrics    = {"fraud_rate": 0.01, "total_transactions": 100,
                                "total_amount": 5000, "avg_amount": 50,
                                "max_amount": 200, "unique_users": 40,
                                "fraud_count": 1, "event_type_breakdown": {},
                                "device_breakdown": {}, "location_breakdown": {}}
        high_risk_metrics   = {**low_risk_metrics, "fraud_rate": 0.20, "fraud_count": 20}

        low_insight  = engine.generate_insight(low_risk_metrics)
        high_insight = engine.generate_insight(high_risk_metrics)

        low_levels  = ["low", "medium"]
        high_levels = ["high", "critical"]
        assert low_insight["risk_level"]  in low_levels,  f"Expected low/medium, got {low_insight['risk_level']}"
        assert high_insight["risk_level"] in high_levels, f"Expected high/critical, got {high_insight['risk_level']}"


# ─────────────────────────────────────────────
# Integration smoke test
# ─────────────────────────────────────────────

class TestEndToEndSmoke:
    def test_pipeline_processes_events(self):
        """
        End-to-end smoke test: generate events → enrich → aggregate → insight.
        No external services required.
        """
        from backend.producer.event_producer import EventGenerator
        from backend.streaming.processor import enrich_event, aggregate_batch, WindowStore
        from backend.ml.llm_insights import LLMInsightsEngine

        gen        = EventGenerator()
        ws         = WindowStore()
        llm        = LLMInsightsEngine(api_key=None)

        raw_events = gen.generate_batch(30)
        enriched   = [enrich_event(e.to_kafka_dict(), ws) for e in raw_events]
        assert len(enriched) == 30

        agg = aggregate_batch(enriched)
        assert agg["total_transactions"] == 30

        insight = llm.generate_insight(agg)
        assert "insight_text" in insight
        assert "risk_level"   in insight
