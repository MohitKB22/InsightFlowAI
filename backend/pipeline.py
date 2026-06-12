"""
Pipeline Orchestrator — ties together producer, streaming processor,
ML inference, LLM insights, alerting, and storage into one runnable process.
Can run all components or individual ones via CLI flags.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("orchestrator")


# ─────────────────────────────────────────────
# Local imports (lazy to survive missing deps)
# ─────────────────────────────────────────────

def _import_components():
    from backend.producer.event_producer import EventGenerator, EventProducer, create_topics
    from backend.streaming.processor import StreamProcessor, WindowStore, enrich_event, aggregate_batch
    from backend.ml.train import InferenceEngine, train_and_save
    from backend.ml.llm_insights import LLMInsightsEngine
    from backend.alerts.alert_engine import AlertEngine
    from backend.storage.storage import PostgresStorage, RedisStorage, S3Storage
    return {
        "EventGenerator": EventGenerator,
        "EventProducer": EventProducer,
        "create_topics": create_topics,
        "StreamProcessor": StreamProcessor,
        "WindowStore": WindowStore,
        "enrich_event": enrich_event,
        "aggregate_batch": aggregate_batch,
        "InferenceEngine": InferenceEngine,
        "train_and_save": train_and_save,
        "LLMInsightsEngine": LLMInsightsEngine,
        "AlertEngine": AlertEngine,
        "PostgresStorage": PostgresStorage,
        "RedisStorage": RedisStorage,
        "S3Storage": S3Storage,
    }


# ─────────────────────────────────────────────
# In-process pipeline (no Kafka required)
# ─────────────────────────────────────────────

class InProcessPipeline:
    """
    Runs the full pipeline in a single process without Kafka.
    Ideal for local development, demos, and CI/testing.

    Flow:
      EventGenerator → enrich_event → InferenceEngine
          → AlertEngine → LLMInsightsEngine (every N events)
          → RedisStorage + PostgresStorage
    """

    def __init__(
        self,
        n_events: int = 500,
        batch_size: int = 50,
        insight_every: int = 100,
        train_model: bool = True,
    ) -> None:
        self.n_events      = n_events
        self.batch_size    = batch_size
        self.insight_every = insight_every
        self.train_model   = train_model

        comps = _import_components()

        # Train / load ML model
        if train_model:
            logger.info("Training ML model…")
            comps["train_and_save"](n_samples=3000)

        self.generator    = comps["EventGenerator"]()
        self.window_store = comps["WindowStore"]()
        self.inference    = comps["InferenceEngine"]()
        self.llm          = comps["LLMInsightsEngine"]()
        self.alerter      = comps["AlertEngine"]()
        self.redis        = comps["RedisStorage"]()
        self.postgres     = comps["PostgresStorage"]()

        self.enrich_event   = comps["enrich_event"]
        self.aggregate_batch = comps["aggregate_batch"]

        try:
            self.inference.load()
        except Exception as exc:
            logger.warning("Inference load failed — predictions disabled: %s", exc)
            self.inference = None

        self.stats = {
            "processed": 0,
            "alerts":    0,
            "insights":  0,
            "fraud_detected": 0,
        }

    def run(self) -> Dict[str, Any]:
        logger.info("InProcessPipeline: processing %d events…", self.n_events)
        batch: List[Dict[str, Any]] = []
        processed_total = 0

        while processed_total < self.n_events:
            # Generate a batch
            chunk_size = min(self.batch_size, self.n_events - processed_total)
            raw_batch  = self.generator.generate_batch(chunk_size)

            enriched_batch: List[Dict[str, Any]] = []
            for raw_event in raw_batch:
                event_dict = raw_event.to_kafka_dict()
                enriched   = self.enrich_event(event_dict, self.window_store)

                # ML inference
                if self.inference:
                    try:
                        pred = self.inference.predict(enriched)
                        enriched.update(pred)
                        if pred.get("is_fraud"):
                            self.stats["fraud_detected"] += 1
                    except Exception as exc:
                        logger.debug("Inference skip: %s", exc)
                        enriched["fraud_probability"] = 0.0
                        enriched["is_fraud"]          = False

                # Alerting
                alerts = self.alerter.evaluate(enriched)
                self.stats["alerts"] += len(alerts)

                # Store to Redis ring buffer
                self.redis.push_to_list("events:recent", enriched, max_len=1000)

                # Store to Postgres (non-blocking)
                self.postgres.insert_processed_event(enriched)

                enriched_batch.append(enriched)
                processed_total += 1

            batch.extend(enriched_batch)
            self.stats["processed"] = processed_total

            # Windowed aggregation + LLM insight every N events
            if processed_total % self.insight_every == 0 or processed_total >= self.n_events:
                agg = self.aggregate_batch(batch[-self.insight_every:])
                if agg:
                    self.redis.store_latest_aggregation(agg)

                    insight = self.llm.generate_insight(agg)
                    insight["insight_id"]   = str(uuid.uuid4())
                    insight["window_start"] = agg.get("window_start", "")
                    insight["window_end"]   = agg.get("window_end", "")
                    insight["raw_metrics"]  = agg

                    self.redis.store_latest_insight(insight)
                    self.postgres.insert_insight(insight)
                    self.stats["insights"] += 1
                    logger.info(
                        "Insight generated [risk=%s]: %s",
                        insight.get("risk_level", "?"),
                        insight.get("insight_text", "")[:80],
                    )

            if processed_total % 100 == 0:
                logger.info(
                    "Progress: %d/%d | Fraud: %d | Alerts: %d | Insights: %d",
                    processed_total, self.n_events,
                    self.stats["fraud_detected"],
                    self.stats["alerts"],
                    self.stats["insights"],
                )

        logger.info("Pipeline complete. Stats: %s", self.stats)
        return self.stats


# ─────────────────────────────────────────────
# Train-only mode
# ─────────────────────────────────────────────

def run_training(n_samples: int = 5000) -> None:
    comps = _import_components()
    logger.info("Training model with %d samples…", n_samples)
    metrics = comps["train_and_save"](n_samples=n_samples)
    print("\n=== Model Training Results ===")
    for k, v in metrics.items():
        print(f"  {k:<25}: {v}")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Data Pipeline — Real-Time ELT with ML & LLM Insights"
    )
    parser.add_argument(
        "mode",
        choices=["pipeline", "train", "producer", "api"],
        help="Component to run",
    )
    parser.add_argument("--events",      type=int, default=500,  help="Number of events to process")
    parser.add_argument("--batch-size",  type=int, default=50,   help="Processing batch size")
    parser.add_argument("--n-samples",   type=int, default=3000, help="Training sample count")
    parser.add_argument("--no-train",    action="store_true",    help="Skip model training in pipeline mode")
    parser.add_argument("--kafka-servers", default="localhost:9092", help="Kafka bootstrap servers")
    args = parser.parse_args()

    if args.mode == "train":
        run_training(n_samples=args.n_samples)

    elif args.mode == "pipeline":
        pipe = InProcessPipeline(
            n_events    = args.events,
            batch_size  = args.batch_size,
            train_model = not args.no_train,
        )
        stats = pipe.run()
        print("\n=== Pipeline Execution Summary ===")
        for k, v in stats.items():
            print(f"  {k:<20}: {v}")

    elif args.mode == "producer":
        comps = _import_components()
        comps["create_topics"](args.kafka_servers)
        gen      = comps["EventGenerator"]()
        producer = comps["EventProducer"](args.kafka_servers)
        logger.info("Kafka producer running — Ctrl+C to stop.")
        count = 0
        try:
            while True:
                event = gen.generate_event()
                producer.publish(event)
                count += 1
                if count % 50 == 0:
                    logger.info("Published %d events.", count)
                time.sleep(0.1)
        except KeyboardInterrupt:
            producer.close()
            logger.info("Producer stopped. Total: %d", count)

    elif args.mode == "api":
        import uvicorn
        uvicorn.run(
            "backend.api.main:app",
            host="0.0.0.0",
            port=int(os.getenv("API_PORT", 8080)),
            reload=True,
        )


if __name__ == "__main__":
    main()
