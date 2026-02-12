"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


# ── Training Config ──────────────────────────────────────

class TrainingConfig(BaseModel):
    num_epochs: int = Field(default=3, ge=1, le=100)
    learning_rate: float = Field(default=2e-4, gt=0, le=1.0)
    batch_size: int = Field(default=4, ge=1, le=128)
    max_seq_length: int = Field(default=512, ge=64, le=8192)
    lora_r: int = Field(default=16, ge=4, le=256)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    warmup_steps: int = Field(default=10, ge=0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    fp16: bool = True
    lora_target_modules: Optional[List[str]] = None  # Auto-detected if None


# ── Job Schemas ──────────────────────────────────────────

class JobCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    base_model: str = Field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    dataset_id: str
    config: TrainingConfig = TrainingConfig()


class JobResponse(BaseModel):
    id: str
    name: str
    status: str
    base_model: str
    dataset_id: str
    config: Dict[str, Any]
    current_epoch: int
    total_epochs: int
    current_step: int
    total_steps: int
    train_loss: Optional[float]
    eval_loss: Optional[float]
    metrics: Dict[str, Any]
    output_model_id: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class JobList(BaseModel):
    jobs: List[JobResponse]
    total: int


# ── Model Schemas ────────────────────────────────────────

class ModelResponse(BaseModel):
    id: str
    name: str
    base_model: str
    job_id: str
    status: str
    adapter_path: str
    metrics: Dict[str, Any]
    config: Dict[str, Any]
    size_bytes: int
    created_at: datetime

    class Config:
        from_attributes = True


class ModelList(BaseModel):
    models: List[ModelResponse]
    total: int


# ── Inference Schemas ────────────────────────────────────

class InferenceRequest(BaseModel):
    model_id: str
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    system_prompt: Optional[str] = None


class InferenceResponse(BaseModel):
    model_id: str
    prompt: str
    generated_text: str
    tokens_generated: int
    latency_ms: float


# ── Dataset Schemas ──────────────────────────────────────

class DatasetResponse(BaseModel):
    id: str
    name: str
    file_path: str
    file_size: int
    num_samples: int
    format: str
    created_at: datetime

    class Config:
        from_attributes = True


class DatasetList(BaseModel):
    datasets: List[DatasetResponse]
    total: int
