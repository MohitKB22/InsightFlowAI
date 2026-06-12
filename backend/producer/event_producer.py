"""
Kafka Event Producer — generates realistic synthetic transaction events
and publishes them to the raw_events topic.
"""
from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from faker import Faker

# Use kafka-python (simpler, no C dependency)
try:
    from kafka import KafkaProducer as _KafkaProducer
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka.errors import TopicAlreadyExistsError, NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from backend.models import DeviceType, EventType, RawEvent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

fake = Faker()
Faker.seed(42)
random.seed(42)

# ─────────────────────────────────────────────
# Topic configuration
# ─────────────────────────────────────────────

TOPICS_CONFIG = [
    {"name": "raw_events",        "partitions": 6, "replication_factor": 1},
    {"name": "processed_events",  "partitions": 6, "replication_factor": 1},
    {"name": "aggregated_metrics","partitions": 3, "replication_factor": 1},
    {"name": "alerts",            "partitions": 3, "replication_factor": 1},
    {"name": "business_insights", "partitions": 2, "replication_factor": 1},
]

# Fraud patterns: some users are "suspicious"
SUSPICIOUS_USER_IDS = [f"user_{i:04d}" for i in range(1, 11)]
NORMAL_USER_IDS     = [f"user_{i:04d}" for i in range(11, 201)]
ALL_USER_IDS        = SUSPICIOUS_USER_IDS + NORMAL_USER_IDS

LOCATIONS = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "Mumbai",
    "London", "Dubai", "Singapore", "Tokyo", "Sydney",
]


# ─────────────────────────────────────────────
# Synthetic event generator
# ─────────────────────────────────────────────

class EventGenerator:
    """Generates realistic transaction events with embedded fraud patterns."""

    def __init__(self) -> None:
        self._session_map: Dict[str, str] = {}

    def _get_session(self, user_id: str) -> str:
        if user_id not in self._session_map or random.random() < 0.1:
            self._session_map[user_id] = fake.uuid4()
        return self._session_map[user_id]

    def generate_normal_event(self, user_id: str) -> RawEvent:
        return RawEvent(
            user_id=user_id,
            session_id=self._get_session(user_id),
            transaction_amount=round(random.uniform(1, 500), 2),
            location=random.choice(LOCATIONS),
            device_type=random.choice(list(DeviceType)),
            event_type=random.choices(
                list(EventType),
                weights=[40, 15, 10, 20, 5, 5, 5],
            )[0],
            ip_address=fake.ipv4(),
            merchant_id=f"merchant_{random.randint(1, 50):04d}",
        )

    def generate_fraudulent_event(self, user_id: str) -> RawEvent:
        """High-amount, foreign-location, API device pattern."""
        return RawEvent(
            user_id=user_id,
            session_id=self._get_session(user_id),
            transaction_amount=round(random.uniform(5000, 50000), 2),
            location=random.choice(["Lagos", "Unknown", "Offshore", "VPN"]),
            device_type=DeviceType.API,
            event_type=random.choice([EventType.TRANSFER, EventType.WITHDRAWAL]),
            ip_address=fake.ipv4(),
            merchant_id=f"merchant_{random.randint(500, 999):04d}",
        )

    def generate_event(self) -> RawEvent:
        """Choose user and whether to generate fraud or normal event."""
        if random.random() < 0.08:           # 8% fraud rate
            user_id = random.choice(SUSPICIOUS_USER_IDS)
            return self.generate_fraudulent_event(user_id)
        user_id = random.choice(ALL_USER_IDS)
        return self.generate_normal_event(user_id)

    def generate_batch(self, size: int = 50) -> List[RawEvent]:
        return [self.generate_event() for _ in range(size)]


# ─────────────────────────────────────────────
# Kafka topic admin
# ─────────────────────────────────────────────

def create_topics(bootstrap_servers: str = "localhost:9092") -> None:
    """Create Kafka topics if they don't exist."""
    if not KAFKA_AVAILABLE:
        logger.warning("kafka-python not installed – skipping topic creation")
        return
    try:
        admin = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            request_timeout_ms=5000,
        )
        new_topics = [
            NewTopic(
                name=t["name"],
                num_partitions=t["partitions"],
                replication_factor=t["replication_factor"],
            )
            for t in TOPICS_CONFIG
        ]
        admin.create_topics(new_topics=new_topics, validate_only=False)
        logger.info("Topics created: %s", [t["name"] for t in TOPICS_CONFIG])
    except TopicAlreadyExistsError:
        logger.info("Topics already exist.")
    except NoBrokersAvailable:
        logger.warning("Kafka broker not reachable — skipping topic creation.")
    except Exception as exc:
        logger.warning("Topic creation error: %s", exc)


# ─────────────────────────────────────────────
# Kafka producer
# ─────────────────────────────────────────────

class EventProducer:
    """Wraps KafkaProducer for event publishing."""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "raw_events",
    ) -> None:
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.producer: Optional[Any] = None

        if KAFKA_AVAILABLE:
            try:
                self.producer = _KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    key_serializer=lambda k: k.encode("utf-8") if k else None,
                    acks="all",
                    retries=3,
                    compression_type="gzip",
                    linger_ms=10,
                    batch_size=32_768,
                )
                logger.info("KafkaProducer connected to %s", bootstrap_servers)
            except NoBrokersAvailable:
                logger.warning("Kafka not available — running in dry-run mode.")
                self.producer = None
        else:
            logger.warning("kafka-python not installed — dry-run mode.")

    def publish(self, event: RawEvent) -> None:
        """Publish a single event to Kafka."""
        data = event.to_kafka_dict()
        if self.producer:
            future = self.producer.send(
                self.topic,
                key=event.user_id,
                value=data,
            )
            future.get(timeout=10)
        else:
            logger.debug("DRY-RUN | %s", json.dumps(data))

    def publish_batch(self, events: List[RawEvent]) -> None:
        for event in events:
            self.publish(event)
        if self.producer:
            self.producer.flush()

    def close(self) -> None:
        if self.producer:
            self.producer.close()


# ─────────────────────────────────────────────
# Main streaming loop
# ─────────────────────────────────────────────

def run_producer(
    bootstrap_servers: str = "localhost:9092",
    topic: str = "raw_events",
    events_per_second: int = 10,
    max_events: Optional[int] = None,
) -> None:
    """
    Continuously generate and publish events at the given rate.

    Args:
        bootstrap_servers: Kafka broker address.
        topic: Target Kafka topic.
        events_per_second: Target throughput (approximate).
        max_events: Stop after this many events (None = run forever).
    """
    create_topics(bootstrap_servers)

    generator = EventGenerator()
    producer  = EventProducer(bootstrap_servers=bootstrap_servers, topic=topic)
    interval  = 1.0 / max(events_per_second, 1)

    total_sent = 0
    logger.info(
        "Producer started — target: %d events/s on topic '%s'",
        events_per_second, topic,
    )

    try:
        while True:
            t0    = time.perf_counter()
            event = generator.generate_event()
            producer.publish(event)
            total_sent += 1

            if total_sent % 100 == 0:
                logger.info("Published %d events so far.", total_sent)

            if max_events and total_sent >= max_events:
                logger.info("Reached max_events limit (%d). Stopping.", max_events)
                break

            elapsed = time.perf_counter() - t0
            sleep_for = max(0.0, interval - elapsed)
            time.sleep(sleep_for)

    except KeyboardInterrupt:
        logger.info("Producer interrupted by user.")
    finally:
        producer.close()
        logger.info("Producer closed. Total events sent: %d", total_sent)


if __name__ == "__main__":
    run_producer(events_per_second=5, max_events=200)
