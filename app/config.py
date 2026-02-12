"""
Application configuration — environment-driven with sensible defaults.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────
    APP_NAME: str = "LLM Fine-Tuning Platform"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # ── Database ─────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./data/finetune.db"

    # ── Redis / Celery ───────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ── Storage ──────────────────────────────────────────
    STORAGE_BACKEND: str = "local"  # "local" or "minio"
    STORAGE_PATH: str = "./data"

    # MinIO (S3-compatible)
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "finetune"
    MINIO_SECURE: bool = False

    # ── Training ─────────────────────────────────────────
    DEFAULT_BASE_MODEL: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    MAX_SEQ_LENGTH: int = 512
    DEFAULT_LORA_R: int = 16
    DEFAULT_LORA_ALPHA: int = 32
    DEFAULT_LORA_DROPOUT: float = 0.05
    DEFAULT_LEARNING_RATE: float = 2e-4
    DEFAULT_NUM_EPOCHS: int = 3
    DEFAULT_BATCH_SIZE: int = 4
    DEFAULT_WARMUP_STEPS: int = 10
    DEFAULT_WEIGHT_DECAY: float = 0.01
    GRADIENT_ACCUMULATION_STEPS: int = 4

    # ── HuggingFace ──────────────────────────────────────
    HF_TOKEN: Optional[str] = None
    HF_CACHE_DIR: str = "./data/hf_cache"

    # ── Inference ────────────────────────────────────────
    INFERENCE_BACKEND: str = "transformers"  # "transformers" or "vllm"
    DEFAULT_MAX_NEW_TOKENS: int = 256
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_TOP_P: float = 0.9

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
