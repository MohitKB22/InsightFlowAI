"""
Shared Pydantic models and schemas used across the pipeline.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class EventType(str, Enum):
    PURCHASE = "purchase"
    TRANSFER = "transfer"
    WITHDRAWAL = "withdrawal"
    LOGIN = "login"
    FAILED_LOGIN = "failed_login"
    PROFILE_UPDATE = "profile_update"
    PASSWORD_CHANGE = "password_change"


class DeviceType(str, Enum):
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"
    API = "api"


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    FRAUD_DETECTED = "fraud_detected"
    HIGH_TRANSACTION = "high_transaction"
    FAILED_ATTEMPTS = "failed_attempts"
    ANOMALY = "anomaly"
    INSIGHT = "business_insight"


# ─────────────────────────────────────────────
# Raw Event Schema (Kafka message)
# ─────────────────────────────────────────────

class RawEvent(BaseModel):
    """Schema for raw streaming events from Kafka producer."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    transaction_amount: float
    location: str
    device_type: DeviceType
    event_type: EventType
    ip_address: Optional[str] = None
    merchant_id: Optional[str] = None
    currency: str = "USD"
    metadata: Optional[Dict[str, Any]] = None

    @field_validator("transaction_amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("transaction_amount must be non-negative")
        return round(v, 2)

    def to_kafka_dict(self) -> Dict[str, Any]:
        """Serialize for Kafka message."""
        data = self.model_dump()
        data["timestamp"] = self.timestamp.isoformat()
        data["event_type"] = self.event_type.value
        data["device_type"] = self.device_type.value
        return data


# ─────────────────────────────────────────────
# Processed Event (after streaming layer)
# ─────────────────────────────────────────────

class ProcessedEvent(BaseModel):
    """Enriched event after PySpark streaming transforms."""
    event_id: str
    user_id: str
    session_id: str
    timestamp: datetime
    transaction_amount: float
    location: str
    device_type: str
    event_type: str
    # Derived features
    hour_of_day: int
    day_of_week: int
    is_weekend: bool
    is_night: bool
    # Window aggregates
    txn_count_1h: int = 0
    avg_amount_1h: float = 0.0
    max_amount_1h: float = 0.0
    failed_attempts_1h: int = 0
    unique_locations_1h: int = 0
    # ML prediction
    fraud_score: Optional[float] = None
    is_fraud: Optional[bool] = None
    processed_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# Aggregated Metrics (Redis / dashboard)
# ─────────────────────────────────────────────

class AggregatedMetrics(BaseModel):
    """Window-level aggregation metrics."""
    window_start: datetime
    window_end: datetime
    total_transactions: int
    total_amount: float
    avg_amount: float
    max_amount: float
    min_amount: float
    unique_users: int
    fraud_count: int
    fraud_rate: float
    event_type_breakdown: Dict[str, int] = {}
    device_breakdown: Dict[str, int] = {}
    location_breakdown: Dict[str, int] = {}
    computed_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# Business Insight (LLM output)
# ─────────────────────────────────────────────

class BusinessInsight(BaseModel):
    """LLM-generated insight from aggregated metrics."""
    insight_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    window_start: datetime
    window_end: datetime
    insight_text: str
    key_findings: List[str] = []
    recommendations: List[str] = []
    risk_level: str = "low"
    model_used: str = "gpt-4o-mini"
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    raw_metrics: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────
# Alert Schema
# ─────────────────────────────────────────────

class Alert(BaseModel):
    """Alert triggered by rule engine."""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    user_id: Optional[str] = None
    event_id: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    resolved: bool = False
    metadata: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────
# ML Prediction
# ─────────────────────────────────────────────

class MLPrediction(BaseModel):
    """ML model inference result."""
    event_id: str
    user_id: str
    fraud_probability: float
    is_fraud: bool
    confidence: float
    feature_importances: Optional[Dict[str, float]] = None
    model_version: str
    predicted_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# API Response wrappers
# ─────────────────────────────────────────────

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None


class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
