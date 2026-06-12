"""
Storage Layer — PostgreSQL (processed events), Redis (aggregations),
S3 (raw event archive), Elasticsearch (predictions + search).
Each class is independently usable with graceful fallback.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PostgreSQL Storage
# ─────────────────────────────────────────────

class PostgresStorage:
    """
    Handles writes/reads for processed events, alerts, and insights
    using SQLAlchemy Core (no ORM dependency).
    """

    DDL = """
    CREATE TABLE IF NOT EXISTS processed_events (
        id              SERIAL PRIMARY KEY,
        event_id        VARCHAR(64) UNIQUE NOT NULL,
        user_id         VARCHAR(64) NOT NULL,
        session_id      VARCHAR(64),
        event_time      TIMESTAMPTZ NOT NULL,
        event_type      VARCHAR(32),
        device_type     VARCHAR(32),
        location        VARCHAR(128),
        transaction_amount NUMERIC(18,2),
        fraud_probability  NUMERIC(6,4),
        is_fraud           BOOLEAN DEFAULT FALSE,
        hour_of_day        SMALLINT,
        day_of_week        SMALLINT,
        txn_count_1h       INTEGER,
        failed_attempts_1h INTEGER,
        processed_at    TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_proc_events_user  ON processed_events(user_id);
    CREATE INDEX IF NOT EXISTS idx_proc_events_time  ON processed_events(event_time);
    CREATE INDEX IF NOT EXISTS idx_proc_events_fraud ON processed_events(is_fraud);

    CREATE TABLE IF NOT EXISTS alerts (
        id           SERIAL PRIMARY KEY,
        alert_id     VARCHAR(64) UNIQUE NOT NULL,
        alert_type   VARCHAR(64),
        severity     VARCHAR(16),
        title        TEXT,
        description  TEXT,
        user_id      VARCHAR(64),
        event_id     VARCHAR(64),
        value        NUMERIC(18,2),
        triggered_at TIMESTAMPTZ DEFAULT NOW(),
        resolved     BOOLEAN DEFAULT FALSE
    );

    CREATE TABLE IF NOT EXISTS business_insights (
        id           SERIAL PRIMARY KEY,
        insight_id   VARCHAR(64) UNIQUE NOT NULL,
        window_start TIMESTAMPTZ,
        window_end   TIMESTAMPTZ,
        insight_text TEXT,
        key_findings JSONB,
        recommendations JSONB,
        risk_level   VARCHAR(16),
        model_used   VARCHAR(64),
        generated_at TIMESTAMPTZ DEFAULT NOW(),
        raw_metrics  JSONB
    );
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        self._url    = db_url or os.getenv("DATABASE_URL", "")
        self._engine = None
        self._connected = False
        self._connect()

    def _connect(self) -> None:
        if not self._url:
            logger.warning("PostgresStorage: no DATABASE_URL — running in no-op mode.")
            return
        try:
            from sqlalchemy import create_engine, text
            self._engine    = create_engine(self._url, pool_pre_ping=True)
            self._text      = text
            with self._engine.connect() as conn:
                conn.execute(text(self.DDL))
                conn.commit()
            self._connected = True
            logger.info("PostgresStorage connected.")
        except Exception as exc:
            logger.warning("PostgresStorage connection failed: %s", exc)

    def insert_processed_event(self, event: Dict[str, Any]) -> bool:
        if not self._connected:
            return False
        sql = self._text("""
            INSERT INTO processed_events
                (event_id, user_id, session_id, event_time, event_type,
                 device_type, location, transaction_amount, fraud_probability,
                 is_fraud, hour_of_day, day_of_week, txn_count_1h, failed_attempts_1h)
            VALUES
                (:event_id, :user_id, :session_id, :event_time, :event_type,
                 :device_type, :location, :transaction_amount, :fraud_probability,
                 :is_fraud, :hour_of_day, :day_of_week, :txn_count_1h, :failed_attempts_1h)
            ON CONFLICT (event_id) DO NOTHING
        """)
        try:
            with self._engine.connect() as conn:
                conn.execute(sql, {
                    "event_id":            event.get("event_id", ""),
                    "user_id":             event.get("user_id", ""),
                    "session_id":          event.get("session_id", ""),
                    "event_time":          event.get("timestamp", datetime.utcnow()),
                    "event_type":          event.get("event_type", ""),
                    "device_type":         event.get("device_type", ""),
                    "location":            event.get("location", ""),
                    "transaction_amount":  event.get("transaction_amount", 0),
                    "fraud_probability":   event.get("fraud_probability", 0),
                    "is_fraud":            bool(event.get("is_fraud", False)),
                    "hour_of_day":         event.get("hour_of_day", 0),
                    "day_of_week":         event.get("day_of_week", 0),
                    "txn_count_1h":        event.get("txn_count_1h", 0),
                    "failed_attempts_1h":  event.get("failed_attempts_1h", 0),
                })
                conn.commit()
            return True
        except Exception as exc:
            logger.error("PostgresStorage insert_processed_event: %s", exc)
            return False

    def insert_alert(self, alert: Dict[str, Any]) -> bool:
        if not self._connected:
            return False
        sql = self._text("""
            INSERT INTO alerts (alert_id, alert_type, severity, title, description,
                                user_id, event_id, value, triggered_at)
            VALUES (:alert_id, :alert_type, :severity, :title, :description,
                    :user_id, :event_id, :value, :triggered_at)
            ON CONFLICT (alert_id) DO NOTHING
        """)
        try:
            with self._engine.connect() as conn:
                conn.execute(sql, alert)
                conn.commit()
            return True
        except Exception as exc:
            logger.error("PostgresStorage insert_alert: %s", exc)
            return False

    def insert_insight(self, insight: Dict[str, Any]) -> bool:
        if not self._connected:
            return False
        sql = self._text("""
            INSERT INTO business_insights
                (insight_id, window_start, window_end, insight_text,
                 key_findings, recommendations, risk_level, model_used, raw_metrics)
            VALUES
                (:insight_id, :window_start, :window_end, :insight_text,
                 :key_findings, :recommendations, :risk_level, :model_used, :raw_metrics)
            ON CONFLICT (insight_id) DO NOTHING
        """)
        try:
            with self._engine.connect() as conn:
                conn.execute(sql, {
                    **insight,
                    "key_findings":    json.dumps(insight.get("key_findings", [])),
                    "recommendations": json.dumps(insight.get("recommendations", [])),
                    "raw_metrics":     json.dumps(insight.get("raw_metrics", {})),
                })
                conn.commit()
            return True
        except Exception as exc:
            logger.error("PostgresStorage insert_insight: %s", exc)
            return False

    def query_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        if not self._connected:
            return []
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    self._text("SELECT * FROM processed_events ORDER BY processed_at DESC LIMIT :lim"),
                    {"lim": limit},
                ).mappings().all()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("PostgresStorage query: %s", exc)
            return []


# ─────────────────────────────────────────────
# Redis Storage (aggregations cache)
# ─────────────────────────────────────────────

class RedisStorage:
    """
    Caches windowed aggregation metrics for fast dashboard reads.
    Uses JSON-serialised values with TTL.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        ttl: int = 3600,
    ) -> None:
        self._ttl = ttl
        self._client = None
        self._connected = False
        try:
            import redis
            self._client    = redis.Redis(host=host, port=port, db=db,
                                          decode_responses=True, socket_connect_timeout=3)
            self._client.ping()
            self._connected = True
            logger.info("RedisStorage connected at %s:%d.", host, port)
        except Exception as exc:
            logger.warning("RedisStorage not available: %s — using in-memory fallback.", exc)
            self._memory: Dict[str, Any] = {}

    def set_metrics(self, key: str, metrics: Dict[str, Any], ttl: Optional[int] = None) -> None:
        value = json.dumps(metrics, default=str)
        expiry = ttl or self._ttl
        if self._connected and self._client:
            self._client.setex(key, expiry, value)
        else:
            self._memory[key] = metrics

    def get_metrics(self, key: str) -> Optional[Dict[str, Any]]:
        if self._connected and self._client:
            raw = self._client.get(key)
            return json.loads(raw) if raw else None
        return self._memory.get(key)

    def increment(self, key: str, amount: int = 1) -> int:
        if self._connected and self._client:
            return int(self._client.incrby(key, amount))
        current = int(self._memory.get(key, 0))
        self._memory[key] = current + amount
        return current + amount

    def push_to_list(self, key: str, value: Any, max_len: int = 1000) -> None:
        """Append to a Redis list (ring buffer for recent events)."""
        serialized = json.dumps(value, default=str)
        if self._connected and self._client:
            self._client.rpush(key, serialized)
            self._client.ltrim(key, -max_len, -1)
        else:
            lst = self._memory.setdefault(key, [])
            lst.append(value)
            if len(lst) > max_len:
                self._memory[key] = lst[-max_len:]

    def get_list(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        if self._connected and self._client:
            raw_list = self._client.lrange(key, start, end)
            return [json.loads(r) for r in raw_list]
        return self._memory.get(key, [])

    def store_latest_aggregation(self, metrics: Dict[str, Any]) -> None:
        self.set_metrics("metrics:latest", metrics)
        self.push_to_list("metrics:history", metrics, max_len=500)

    def store_latest_insight(self, insight: Dict[str, Any]) -> None:
        self.set_metrics("insight:latest", insight)
        self.push_to_list("insight:history", insight, max_len=100)

    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        return self.get_metrics("metrics:latest")

    def get_metrics_history(self, n: int = 60) -> List[Dict[str, Any]]:
        return self.get_list("metrics:history", -n, -1)

    def get_latest_insight(self) -> Optional[Dict[str, Any]]:
        return self.get_metrics("insight:latest")


# ─────────────────────────────────────────────
# S3 Storage (raw event archive)
# ─────────────────────────────────────────────

class S3Storage:
    """
    Archives raw events to S3 using partitioned paths:
    s3://<bucket>/year=YYYY/month=MM/day=DD/hour=HH/<batch_id>.json
    """

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
    ) -> None:
        self.bucket  = bucket
        self._client = None
        self._connected = False
        try:
            import boto3
            session = boto3.Session(
                aws_access_key_id     = aws_access_key or os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key = aws_secret_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name           = region,
            )
            self._client    = session.client("s3")
            self._connected = True
            logger.info("S3Storage connected (bucket=%s).", bucket)
        except Exception as exc:
            logger.warning("S3Storage not available: %s", exc)

    def _partition_key(self, dt: Optional[datetime] = None) -> str:
        dt = dt or datetime.utcnow()
        return f"year={dt.year}/month={dt.month:02d}/day={dt.day:02d}/hour={dt.hour:02d}"

    def archive_batch(self, events: List[Dict[str, Any]], batch_id: Optional[str] = None) -> str:
        import uuid as _uuid
        key = f"{self._partition_key()}/{batch_id or _uuid.uuid4()}.json"
        payload = json.dumps(events, default=str).encode("utf-8")
        if self._connected and self._client:
            try:
                self._client.put_object(Bucket=self.bucket, Key=key, Body=payload,
                                        ContentType="application/json")
                logger.info("S3: archived %d events → s3://%s/%s", len(events), self.bucket, key)
            except Exception as exc:
                logger.error("S3 upload error: %s", exc)
        else:
            logger.debug("S3 dry-run: would upload %d events to s3://%s/%s",
                         len(events), self.bucket, key)
        return key


# ─────────────────────────────────────────────
# Elasticsearch Storage (predictions + full-text search)
# ─────────────────────────────────────────────

class ElasticsearchStorage:
    """Indexes processed events and insights into Elasticsearch."""

    def __init__(self, host: str = "localhost", port: int = 9200) -> None:
        self._client    = None
        self._connected = False
        try:
            from elasticsearch import Elasticsearch
            self._client    = Elasticsearch([f"http://{host}:{port}"],
                                            request_timeout=5)
            if self._client.ping():
                self._connected = True
                self._ensure_indices()
                logger.info("ElasticsearchStorage connected at %s:%d.", host, port)
            else:
                logger.warning("Elasticsearch ping failed.")
        except Exception as exc:
            logger.warning("ElasticsearchStorage not available: %s", exc)

    def _ensure_indices(self) -> None:
        events_mapping = {
            "mappings": {"properties": {
                "event_id":           {"type": "keyword"},
                "user_id":            {"type": "keyword"},
                "event_time":         {"type": "date"},
                "transaction_amount": {"type": "float"},
                "fraud_probability":  {"type": "float"},
                "is_fraud":           {"type": "boolean"},
                "location":           {"type": "keyword"},
                "device_type":        {"type": "keyword"},
                "event_type":         {"type": "keyword"},
            }}
        }
        for idx, mapping in [("events", events_mapping), ("insights", {})]:
            if not self._client.indices.exists(index=idx):
                self._client.indices.create(index=idx, body=mapping)

    def index_event(self, event: Dict[str, Any]) -> bool:
        if not self._connected:
            return False
        try:
            self._client.index(index="events", id=event.get("event_id"), document=event)
            return True
        except Exception as exc:
            logger.error("ES index_event error: %s", exc)
            return False

    def index_insight(self, insight: Dict[str, Any]) -> bool:
        if not self._connected:
            return False
        try:
            self._client.index(index="insights", id=insight.get("insight_id"), document=insight)
            return True
        except Exception as exc:
            logger.error("ES index_insight error: %s", exc)
            return False

    def search_fraud_events(self, min_score: float = 0.8, size: int = 20) -> List[Dict[str, Any]]:
        if not self._connected:
            return []
        query = {"query": {"range": {"fraud_probability": {"gte": min_score}}},
                 "sort": [{"event_time": "desc"}], "size": size}
        try:
            resp = self._client.search(index="events", body=query)
            return [hit["_source"] for hit in resp["hits"]["hits"]]
        except Exception as exc:
            logger.error("ES search error: %s", exc)
            return []
