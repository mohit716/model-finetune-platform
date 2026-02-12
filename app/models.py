"""
SQLAlchemy ORM models for jobs, models, and datasets.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, JSON, Enum as SAEnum
)
from sqlalchemy.orm import relationship
import enum

from app.database import Base


def generate_uuid():
    return str(uuid.uuid4())


def utcnow():
    return datetime.now(timezone.utc)


# ── Enums ────────────────────────────────────────────────

class JobStatus(str, enum.Enum):
    PENDING = "pending"
    VALIDATING = "validating"
    DOWNLOADING = "downloading"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelStatus(str, enum.Enum):
    READY = "ready"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"


# ── Dataset ──────────────────────────────────────────────

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, default=0)
    num_samples = Column(Integer, default=0)
    format = Column(String, default="jsonl")  # jsonl, csv, parquet
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=utcnow)

    def __repr__(self):
        return f"<Dataset {self.name} ({self.num_samples} samples)>"


# ── Fine-Tuning Job ─────────────────────────────────────

class FineTuneJob(Base):
    __tablename__ = "finetune_jobs"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    status = Column(SAEnum(JobStatus), default=JobStatus.PENDING)

    # Model config
    base_model = Column(String, nullable=False)
    dataset_id = Column(String, nullable=False)

    # Hyperparameters (stored as JSON for flexibility)
    config = Column(JSON, default=dict)

    # Training metrics
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=3)
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    train_loss = Column(Float, nullable=True)
    eval_loss = Column(Float, nullable=True)
    metrics = Column(JSON, default=dict)

    # Output
    output_model_id = Column(String, nullable=True)
    output_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)

    # Celery task tracking
    celery_task_id = Column(String, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<Job {self.name} [{self.status}]>"


# ── Fine-Tuned Model ────────────────────────────────────

class FineTunedModel(Base):
    __tablename__ = "finetuned_models"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    base_model = Column(String, nullable=False)
    job_id = Column(String, nullable=False)
    status = Column(SAEnum(ModelStatus), default=ModelStatus.READY)

    # Storage
    adapter_path = Column(String, nullable=False)    # LoRA adapter location
    merged_path = Column(String, nullable=True)      # Optional merged model

    # Metadata
    metrics = Column(JSON, default=dict)
    config = Column(JSON, default=dict)
    size_bytes = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=utcnow)

    def __repr__(self):
        return f"<Model {self.name} ({self.base_model})>"
