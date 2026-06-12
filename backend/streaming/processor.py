"""
Stream Processing Layer — consumes Kafka events, applies windowed aggregations,
enriches with derived features, runs ML inference, and forwards results.

NOTE: This module provides a pandas-based reference implementation that mirrors
the logic of a PySpark Structured Streaming job. A full PySpark implementation
is provided in spark_streaming.py for cluster deployments.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# In-memory window store (mimics Spark watermark)
# ─────────────────────────────────────────────

class WindowStore:
    """
    Sliding window aggregation store.
    Keeps last N minutes of events per user in memory.
    Thread-safe using a lock.
    """

    def __init__(self, window_minutes: int = 60) -> None:
        self._window_td   = timedelta(minutes=window_minutes)
        self._store: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def add_event(self, user_id: str, event: Dict[str, Any]) -> None:
        ts = event.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        event["_ts"] = ts
        with self._lock:
            self._store[user_id].append(event)
            self._evict_old(user_id, ts)

    def _evict_old(self, user_id: str, now: datetime) -> None:
        cutoff = now - self._window_td
        dq = self._store[user_id]
        while dq and dq[0]["_ts"] < cutoff:
            dq.popleft()

    def get_window(self, user_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._store[user_id])

    def compute_features(self, user_id: str, current_ts: datetime) -> Dict[str, Any]:
        """Compute 1-hour window features for a user."""
        events = self.get_window(user_id)
        if not events:
            return self._empty_features()

        amounts      = [e.get("transaction_amount", 0) for e in events]
        failed_count = sum(1 for e in events if e.get("event_type") == "failed_login")
        locations    = {e.get("location", "") for e in events}

        return {
            "txn_count_1h":        len(events),
            "avg_amount_1h":       round(pd.Series(amounts).mean(), 2),
            "max_amount_1h":       round(max(amounts), 2),
            "failed_attempts_1h":  failed_count,
            "unique_locations_1h": len(locations),
        }

    @staticmethod
    def _empty_features() -> Dict[str, Any]:
        return {
            "txn_count_1h": 0,
            "avg_amount_1h": 0.0,
            "max_amount_1h": 0.0,
            "failed_attempts_1h": 0,
            "unique_locations_1h": 0,
        }


# ─────────────────────────────────────────────
# Feature derivation
# ─────────────────────────────────────────────

def derive_time_features(ts: datetime) -> Dict[str, Any]:
    return {
        "hour_of_day": ts.hour,
        "day_of_week": ts.weekday(),
        "is_weekend":  int(ts.weekday() >= 5),
        "is_night":    int(ts.hour < 6 or ts.hour >= 22),
    }


def enrich_event(
    event: Dict[str, Any],
    window_store: WindowStore,
) -> Dict[str, Any]:
    """Combine raw event with time features + window aggregates."""
    ts = event.get("timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    elif ts is None:
        ts = datetime.utcnow()

    enriched = {**event}
    enriched.update(derive_time_features(ts))

    user_id = event.get("user_id", "unknown")
    window_store.add_event(user_id, event)
    enriched.update(window_store.compute_features(user_id, ts))
    enriched["processed_at"] = datetime.utcnow().isoformat()
    return enriched


# ─────────────────────────────────────────────
# Aggregation (tumbling window over a batch)
# ─────────────────────────────────────────────

def aggregate_batch(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics over a batch of processed events.
    Equivalent to a 1-minute tumbling window in Spark.
    """
    if not events:
        return {}

    df = pd.DataFrame(events)
    amounts = df.get("transaction_amount", pd.Series(dtype=float)).fillna(0)

    fraud_count = int(df.get("is_fraud", pd.Series([False] * len(df))).sum()) \
        if "is_fraud" in df.columns else 0

    result: Dict[str, Any] = {
        "window_start":         datetime.utcnow().isoformat(),
        "window_end":           datetime.utcnow().isoformat(),
        "total_transactions":   len(df),
        "total_amount":         round(float(amounts.sum()), 2),
        "avg_amount":           round(float(amounts.mean()), 2),
        "max_amount":           round(float(amounts.max()), 2),
        "min_amount":           round(float(amounts.min()), 2),
        "unique_users":         int(df["user_id"].nunique()) if "user_id" in df.columns else 0,
        "fraud_count":          fraud_count,
        "fraud_rate":           round(fraud_count / max(len(df), 1), 4),
    }

    # Breakdown by event type
    if "event_type" in df.columns:
        result["event_type_breakdown"] = df["event_type"].value_counts().to_dict()
    if "device_type" in df.columns:
        result["device_breakdown"] = df["device_type"].value_counts().to_dict()
    if "location" in df.columns:
        result["location_breakdown"] = df["location"].value_counts().head(10).to_dict()

    return result


# ─────────────────────────────────────────────
# Stream processor (connects all pieces)
# ─────────────────────────────────────────────

class StreamProcessor:
    """
    Main processing unit:
    - Consumes events from Kafka (or a mock source)
    - Enriches with window features
    - Runs ML inference
    - Publishes results
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        input_topic: str = "raw_events",
        output_topic: str = "processed_events",
        inference_engine: Optional[Any] = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.input_topic       = input_topic
        self.output_topic      = output_topic
        self.window_store      = WindowStore(window_minutes=60)
        self.inference_engine  = inference_engine
        self._consumer: Optional[Any] = None
        self._producer: Optional[Any] = None
        self._running = False

        self._init_kafka()

    def _init_kafka(self) -> None:
        try:
            from kafka import KafkaConsumer, KafkaProducer
            from kafka.errors import NoBrokersAvailable
            self._consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id="stream_processor_group",
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                session_timeout_ms=30_000,
                request_timeout_ms=40_000,
            )
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
            )
            logger.info("Kafka consumer/producer initialized.")
        except Exception as exc:
            logger.warning("Kafka not available (%s) — dry-run mode.", exc)
            self._consumer = None
            self._producer = None

    def process_event(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich + infer on a single raw event dict."""
        enriched = enrich_event(raw_event, self.window_store)

        if self.inference_engine:
            try:
                prediction = self.inference_engine.predict(enriched)
                enriched.update(prediction)
            except Exception as exc:
                logger.warning("Inference error: %s", exc)
                enriched["fraud_probability"] = 0.0
                enriched["is_fraud"]          = False
                enriched["confidence"]        = 0.0

        return enriched

    def _publish(self, event: Dict[str, Any]) -> None:
        if self._producer:
            self._producer.send(self.output_topic, value=event)
        else:
            logger.debug("DRY-RUN processed: %s", event.get("event_id", "?"))

    def run(self, max_messages: Optional[int] = None) -> None:
        """Start consuming and processing in a loop."""
        self._running = True
        processed = 0
        logger.info("StreamProcessor running on topic '%s'…", self.input_topic)

        if self._consumer is None:
            logger.warning("No Kafka consumer — exiting run().")
            return

        try:
            for message in self._consumer:
                if not self._running:
                    break
                raw = message.value
                processed_event = self.process_event(raw)
                self._publish(processed_event)
                processed += 1

                if processed % 50 == 0:
                    logger.info("Processed %d messages.", processed)

                if max_messages and processed >= max_messages:
                    break
        except KeyboardInterrupt:
            logger.info("Processor interrupted.")
        finally:
            self.stop()
            logger.info("Total messages processed: %d", processed)

    def stop(self) -> None:
        self._running = False
        if self._consumer:
            self._consumer.close()
        if self._producer:
            self._producer.flush()
            self._producer.close()


# ─────────────────────────────────────────────
# PySpark Structured Streaming (cluster version)
# ─────────────────────────────────────────────

PYSPARK_JOB = '''
"""
PySpark Structured Streaming Job
Deploy on EMR or local Spark installation.
Run: spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0
        backend/streaming/spark_job.py
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg, col, count, from_json, max as spark_max, min as spark_min,
    sum as spark_sum, to_timestamp, window, udf,
)
from pyspark.sql.types import (
    DoubleType, StringType, StructField, StructType, TimestampType,
)

EVENT_SCHEMA = StructType([
    StructField("event_id",           StringType()),
    StructField("user_id",            StringType()),
    StructField("session_id",         StringType()),
    StructField("timestamp",          StringType()),
    StructField("transaction_amount", DoubleType()),
    StructField("location",           StringType()),
    StructField("device_type",        StringType()),
    StructField("event_type",         StringType()),
])

spark = (
    SparkSession.builder
    .appName("LLMDataPipeline_StreamProcessor")
    .config("spark.sql.shuffle.partitions", "6")
    .config("spark.streaming.stopGracefullyOnShutdown", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

raw_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "raw_events")
    .option("startingOffsets", "latest")
    .option("failOnDataLoss", "false")
    .load()
)

parsed = (
    raw_df
    .select(from_json(col("value").cast("string"), EVENT_SCHEMA).alias("data"))
    .select("data.*")
    .withColumn("event_time", to_timestamp("timestamp"))
    .withWatermark("event_time", "10 minutes")
)

# Tumbling 1-minute window aggregations
windowed_agg = (
    parsed
    .groupBy(window("event_time", "1 minute"))
    .agg(
        count("*").alias("total_transactions"),
        spark_sum("transaction_amount").alias("total_amount"),
        avg("transaction_amount").alias("avg_amount"),
        spark_max("transaction_amount").alias("max_amount"),
    )
)

query = (
    windowed_agg.writeStream
    .outputMode("update")
    .format("console")
    .option("truncate", "false")
    .trigger(processingTime="30 seconds")
    .start()
)

query.awaitTermination()
'''


if __name__ == "__main__":
    # Quick smoke test: process a list of synthetic events
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from backend.producer.event_producer import EventGenerator

    gen = EventGenerator()
    proc = StreamProcessor.__new__(StreamProcessor)
    proc.window_store     = WindowStore()
    proc.inference_engine = None

    batch = gen.generate_batch(20)
    results = []
    for ev in batch:
        processed = proc.process_event(ev.to_kafka_dict())
        results.append(processed)

    df = pd.DataFrame(results)
    print(df[["user_id", "transaction_amount", "hour_of_day",
              "txn_count_1h", "fraud_probability"]].head(10).to_string())
    agg = aggregate_batch(results)
    print("\n=== Batch Aggregation ===")
    for k, v in agg.items():
        if not isinstance(v, dict):
            print(f"  {k:<25}: {v}")
