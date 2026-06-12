"""
FastAPI REST API — exposes pipeline metrics, alerts, insights, and
manual inference endpoints for the dashboard and external consumers.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.alerts.alert_engine import AlertEngine
from backend.ml.train import InferenceEngine
from backend.models import APIResponse
from backend.storage.storage import PostgresStorage, RedisStorage

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="LLM Data Pipeline API",
    description="Real-Time ELT Pipeline with ML Fraud Detection & LLM Insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Shared service singletons (lazy init)
# ─────────────────────────────────────────────

_redis:     Optional[RedisStorage]    = None
_postgres:  Optional[PostgresStorage] = None
_inference: Optional[InferenceEngine] = None
_alerter:   Optional[AlertEngine]     = None


def get_redis() -> RedisStorage:
    global _redis
    if _redis is None:
        _redis = RedisStorage(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
        )
    return _redis


def get_postgres() -> PostgresStorage:
    global _postgres
    if _postgres is None:
        _postgres = PostgresStorage(db_url=os.getenv("DATABASE_URL", ""))
    return _postgres


def get_inference() -> InferenceEngine:
    global _inference
    if _inference is None:
        _inference = InferenceEngine()
        try:
            _inference.load()
        except Exception as exc:
            logger.warning("InferenceEngine load failed: %s", exc)
    return _inference


def get_alerter() -> AlertEngine:
    global _alerter
    if _alerter is None:
        _alerter = AlertEngine(
            slack_token=os.getenv("SLACK_BOT_TOKEN"),
            fraud_threshold=float(os.getenv("ALERT_FRAUD_THRESHOLD", 0.85)),
            high_amount=float(os.getenv("ALERT_TRANSACTION_THRESHOLD", 10000)),
        )
    return _alerter


# ─────────────────────────────────────────────
# Request / response models
# ─────────────────────────────────────────────

class InferenceRequest(BaseModel):
    user_id: str
    transaction_amount: float
    location: str
    device_type: str
    event_type: str
    hour_of_day: Optional[int] = 12
    day_of_week: Optional[int] = 0
    is_weekend: Optional[int] = 0
    is_night: Optional[int] = 0
    txn_count_1h: Optional[int] = 1
    avg_amount_1h: Optional[float] = 0.0
    max_amount_1h: Optional[float] = 0.0
    failed_attempts_1h: Optional[int] = 0
    unique_locations_1h: Optional[int] = 1


# ─────────────────────────────────────────────
# Health & status
# ─────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    }


@app.get("/status", tags=["System"])
def pipeline_status() -> Dict[str, Any]:
    redis = get_redis()
    latest = redis.get_latest_metrics() or {}
    return {
        "pipeline": "running",
        "last_metrics_window": latest.get("window_start"),
        "total_transactions":  latest.get("total_transactions", 0),
        "fraud_rate":          latest.get("fraud_rate", 0),
        "timestamp":           datetime.utcnow().isoformat(),
    }


# ─────────────────────────────────────────────
# Metrics endpoints
# ─────────────────────────────────────────────

@app.get("/metrics/latest", tags=["Metrics"])
def get_latest_metrics() -> Dict[str, Any]:
    redis = get_redis()
    data  = redis.get_latest_metrics()
    if not data:
        return {"message": "No metrics available yet.", "data": {}}
    return {"data": data}


@app.get("/metrics/history", tags=["Metrics"])
def get_metrics_history(n: int = Query(default=60, ge=1, le=500)) -> Dict[str, Any]:
    redis = get_redis()
    history = redis.get_metrics_history(n=n)
    return {"count": len(history), "data": history}


@app.get("/metrics/summary", tags=["Metrics"])
def get_metrics_summary() -> Dict[str, Any]:
    redis   = get_redis()
    history = redis.get_metrics_history(n=60)
    if not history:
        return {"message": "No history available."}

    import statistics
    amounts      = [h.get("avg_amount", 0) for h in history if h.get("avg_amount")]
    fraud_rates  = [h.get("fraud_rate", 0) for h in history]
    txn_counts   = [h.get("total_transactions", 0) for h in history]

    return {
        "windows_analysed":     len(history),
        "avg_transaction_value": round(statistics.mean(amounts), 2) if amounts else 0,
        "avg_fraud_rate":        round(statistics.mean(fraud_rates), 4) if fraud_rates else 0,
        "peak_transactions":     max(txn_counts) if txn_counts else 0,
        "total_transactions":    sum(txn_counts),
    }


# ─────────────────────────────────────────────
# Alerts endpoints
# ─────────────────────────────────────────────

@app.get("/alerts", tags=["Alerts"])
def get_alerts(limit: int = Query(default=50, ge=1, le=500)) -> Dict[str, Any]:
    alerter = get_alerter()
    alerts  = alerter.get_recent_alerts(limit=limit)
    return {"count": len(alerts), "data": alerts}


@app.get("/alerts/counts", tags=["Alerts"])
def get_alert_counts() -> Dict[str, Any]:
    alerter = get_alerter()
    return {"counts": alerter.get_alert_counts()}


# ─────────────────────────────────────────────
# Insights endpoints
# ─────────────────────────────────────────────

@app.get("/insights/latest", tags=["Insights"])
def get_latest_insight() -> Dict[str, Any]:
    redis   = get_redis()
    insight = redis.get_latest_insight()
    if not insight:
        return {"message": "No insights generated yet.", "data": {}}
    return {"data": insight}


@app.get("/insights/history", tags=["Insights"])
def get_insight_history(n: int = Query(default=10, ge=1, le=100)) -> Dict[str, Any]:
    redis   = get_redis()
    history = redis.get_list("insight:history", -n, -1)
    return {"count": len(history), "data": history}


# ─────────────────────────────────────────────
# ML Inference endpoint
# ─────────────────────────────────────────────

@app.post("/inference/predict", tags=["ML"])
def predict_fraud(request: InferenceRequest) -> Dict[str, Any]:
    """
    Run real-time fraud inference on a single transaction event.
    Returns fraud_probability, is_fraud, and confidence score.
    """
    engine = get_inference()
    try:
        result = engine.predict(request.model_dump())
        return {
            "user_id":             request.user_id,
            "transaction_amount":  request.transaction_amount,
            "fraud_probability":   result["fraud_probability"],
            "is_fraud":            result["is_fraud"],
            "confidence":          result["confidence"],
            "predicted_at":        datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        logger.error("Inference endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")


@app.post("/inference/batch", tags=["ML"])
def predict_fraud_batch(requests: List[InferenceRequest]) -> Dict[str, Any]:
    """Batch inference for up to 100 events."""
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Batch size limit is 100.")
    engine  = get_inference()
    results = engine.predict_batch([r.model_dump() for r in requests])
    return {"count": len(results), "predictions": results}


# ─────────────────────────────────────────────
# Events endpoint (from Postgres)
# ─────────────────────────────────────────────

@app.get("/events/recent", tags=["Events"])
def get_recent_events(limit: int = Query(default=50, ge=1, le=500)) -> Dict[str, Any]:
    postgres = get_postgres()
    events   = postgres.query_recent_events(limit=limit)
    # Convert non-serialisable types
    for e in events:
        for k, v in e.items():
            if isinstance(v, datetime):
                e[k] = v.isoformat()
    return {"count": len(events), "data": events}


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8080)),
        reload=True,
        log_level="info",
    )
