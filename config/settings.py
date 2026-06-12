"""
Core configuration module using Pydantic Settings.
Loads from .env file and environment variables.
"""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    app_name: str = "LLM_Data_Pipeline"
    app_env: str = "development"
    log_level: str = "INFO"
    secret_key: str = "dev-secret-key"

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_raw_events: str = "raw_events"
    kafka_topic_processed: str = "processed_events"
    kafka_topic_aggregated: str = "aggregated_metrics"
    kafka_topic_alerts: str = "alerts"
    kafka_topic_insights: str = "business_insights"
    kafka_consumer_group: str = "llm_pipeline_group"
    kafka_num_partitions: int = 6
    kafka_replication_factor: int = 1
    kafka_auto_offset_reset: str = "earliest"
    kafka_enable_auto_commit: bool = False

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "pipeline_db"
    postgres_user: str = "pipeline_user"
    postgres_password: str = "pipeline_pass"
    postgres_pool_size: int = 10

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_ttl: int = 3600

    # Elasticsearch
    es_host: str = "localhost"
    es_port: int = 9200
    es_index_events: str = "events"
    es_index_insights: str = "insights"

    # AWS S3
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    s3_bucket_raw: str = "llm-pipeline-raw-events"
    s3_bucket_processed: str = "llm-pipeline-processed"

    # OpenAI / LLM
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 1500
    llm_temperature: float = 0.3

    # Alerting
    slack_bot_token: Optional[str] = None
    slack_channel: str = "#data-alerts"
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    alert_email_to: Optional[str] = None
    alert_fraud_threshold: float = 0.85
    alert_transaction_threshold: float = 10000.0
    alert_failed_attempts_threshold: int = 5

    # ML Model
    model_path: str = "backend/ml/models/fraud_model.pkl"
    scaler_path: str = "backend/ml/models/scaler.pkl"
    feature_names_path: str = "backend/ml/models/feature_names.pkl"
    model_version: str = "1.0.0"
    inference_batch_size: int = 100

    # Prometheus
    prometheus_port: int = 8000
    metrics_export_interval: int = 15

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_workers: int = 4

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def es_url(self) -> str:
        return f"http://{self.es_host}:{self.es_port}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()


settings = get_settings()
