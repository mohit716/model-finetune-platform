# LLM Fine-Tuning Platform

A self-hosted fine-tuning platform for large language models. Upload datasets, run LoRA/QLoRA training jobs asynchronously, and serve inference — all through a REST API.

Built as a lightweight, open-source alternative to managed services like Amazon SageMaker for fine-tuning workflows.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  FastAPI     │────▶│  Redis       │────▶│  Celery Worker   │
│  REST API    │     │  (broker)    │     │  (GPU training)  │
└─────┬───────┘     └──────────────┘     └────────┬─────────┘
      │                                           │
      ▼                                           ▼
┌─────────────┐                          ┌──────────────────┐
│  SQLite      │                          │  Local Storage /  │
│  (metadata)  │                          │  MinIO (S3)       │
└─────────────┘                          └──────────────────┘
```

**Stack:** FastAPI + Celery + Redis + HuggingFace Transformers + PEFT + TRL

## Features

- **Dataset management** — Upload and validate JSONL datasets (chat or instruction format)
- **Async training** — Submit fine-tuning jobs that run on GPU via Celery workers
- **QLoRA support** — 4-bit quantized training to fit large models on consumer GPUs
- **Chat format support** — Native OpenAI-style `messages` array with assistant-only loss masking
- **Live progress** — Poll job status for step count, loss, and dataset statistics
- **Inference API** — Load fine-tuned models and generate text with chat template support
- **Configurable** — All hyperparameters, LoRA settings, and dataset filters exposed via API

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (T4 16GB or better recommended)
- Redis

### Setup

```bash
git clone <repo-url>
cd model-finetune-platform

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

For gated models (e.g., Llama 3.1), create a `.env` file:

```bash
HF_TOKEN=your_huggingface_token_here
```

### Run

**One command (recommended):**

```bash
./scripts/run-all.sh
```

**Manual start:**

```bash
./scripts/start-redis.sh

celery -A app.workers.celery_app worker --loglevel=info --concurrency=1 &

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at `http://localhost:8000/docs`

## Dataset Support

The platform accepts two JSONL formats. Do not mix formats within a single file.

### Chat format (OpenAI-style)

Each line contains a `messages` array with exactly three roles: `system`, `user`, `assistant`.

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain recursion."},
    {"role": "assistant", "content": "Recursion is when a function calls itself..."}
  ],
  "metadata": {"optional": "ignored by trainer"}
}
```

- Roles must be exactly `["system", "user", "assistant"]` in that order
- Assistant content must be non-empty
- `metadata` field is allowed but ignored during training
- Loss is masked so only assistant tokens contribute (prompt tokens are not penalized)

### Instruction format

```json
{"instruction": "Summarize the following", "input": "Long text here...", "output": "Summary here"}
```

### Format auto-detection

Set `dataset_format: "auto"` (default) in the job config and the platform detects the format from the first line.

## API Endpoints

### Datasets

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/datasets/upload` | Upload a JSONL dataset |
| `GET` | `/api/v1/datasets/` | List all datasets |
| `GET` | `/api/v1/datasets/{id}` | Get dataset details |
| `DELETE` | `/api/v1/datasets/{id}` | Delete a dataset |

### Jobs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/jobs/` | Submit a fine-tuning job |
| `GET` | `/api/v1/jobs/` | List all jobs (filterable by status) |
| `GET` | `/api/v1/jobs/{id}` | Get job status, progress, and metrics |
| `DELETE` | `/api/v1/jobs/{id}` | Cancel a running job |

### Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/models/` | List fine-tuned models |
| `GET` | `/api/v1/models/{id}` | Get model details and metrics |
| `DELETE` | `/api/v1/models/{id}` | Delete a model |

### Inference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/inference/` | Generate text with a fine-tuned model |
| `DELETE` | `/api/v1/inference/cache` | Clear model cache to free GPU memory |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check (GPU status, version) |
| `GET` | `/metrics` | Prometheus metrics |

## Usage Example

### 1. Upload a dataset

```bash
curl -X POST http://localhost:8000/api/v1/datasets/upload \
  -F "file=@train_data.jsonl" \
  -F "name=my-training-data"
```

### 2. Submit a fine-tuning job

```bash
curl -X POST http://localhost:8000/api/v1/jobs/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-finetuned-model",
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",
    "dataset_id": "DATASET_ID_FROM_STEP_1",
    "config": {
      "dataset_format": "chat",
      "max_seq_length": 1024,
      "batch_size": 1,
      "gradient_accumulation_steps": 8,
      "num_epochs": 3,
      "learning_rate": 2e-4,
      "lora_r": 16,
      "lora_alpha": 32,
      "lora_dropout": 0.05,
      "gradient_checkpointing": true
    }
  }'
```

### 3. Monitor training progress

```bash
curl http://localhost:8000/api/v1/jobs/JOB_ID
```

Returns step count, loss, epoch, and dataset statistics (rows validated, dropped, avg/max lengths).

### 4. Run inference

```bash
curl -X POST http://localhost:8000/api/v1/inference/ \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "MODEL_ID_FROM_COMPLETED_JOB",
    "prompt": "Your input text here",
    "system_prompt": "You are a helpful assistant.",
    "max_new_tokens": 256,
    "temperature": 0.3
  }'
```

## Training Configuration

All parameters are optional and have sensible defaults.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_seq_length` | 512 | Maximum token length per training sample |
| `num_epochs` | 3 | Number of training passes over the dataset |
| `learning_rate` | 2e-4 | Learning rate for AdamW optimizer |
| `batch_size` | 4 | Per-device train batch size |
| `gradient_accumulation_steps` | 4 | Effective batch = batch_size x this value |
| `lora_r` | 16 | LoRA rank (higher = more capacity, more memory) |
| `lora_alpha` | 32 | LoRA scaling factor |
| `lora_dropout` | 0.05 | Dropout for regularization |
| `gradient_checkpointing` | true | Trade ~30% speed for ~50% VRAM savings |
| `dataset_format` | "auto" | `"auto"`, `"chat"`, or `"instruction"` |
| `max_user_chars` | null | Drop rows where user content exceeds this |
| `max_assistant_chars` | null | Drop rows where assistant content exceeds this |
| `fp16` | true | Half-precision training |

### Recommended config for T4 (16 GB) with 8B model

```json
{
  "max_seq_length": 1024,
  "batch_size": 1,
  "gradient_accumulation_steps": 8,
  "gradient_checkpointing": true,
  "lora_r": 16,
  "lora_alpha": 32
}
```

## Project Structure

```
model-finetune-platform/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # Settings (env-driven)
│   ├── database.py              # SQLite + SQLAlchemy setup
│   ├── models.py                # DB table definitions
│   ├── schemas.py               # Pydantic request/response models
│   ├── routers/
│   │   ├── datasets.py          # Dataset upload and validation
│   │   ├── jobs.py              # Job submission and management
│   │   ├── models.py            # Trained model listing
│   │   └── inference.py         # Text generation endpoint
│   ├── services/
│   │   └── storage.py           # File storage (local or MinIO/S3)
│   └── workers/
│       ├── celery_app.py        # Celery configuration
│       └── trainer.py           # Training logic (LoRA, chat template, loss masking)
├── scripts/
│   ├── run-all.sh               # Start all services
│   ├── start-redis.sh           # Build and start Redis locally
│   └── test_e2e.py              # End-to-end smoke test
├── data/                        # Runtime data (datasets, models, DB) — gitignored
├── inference_results.md         # Inference output tracking
├── requirements.txt
└── README.md
```

## Validation Case Study: MathForum Mentor Fine-Tuning

The platform was validated by fine-tuning a mentor feedback model using a private mentoring thread archive.

### Dataset

- **Source:** MathForum mentor dataset (private export) — mentor/student math feedback threads
- **Raw data:** ~64,039 thread files, converted to ~75,706 examples
- **Cleaning:** Removed embedded images/base64, empty replies, placeholder text, duplicate threads, reduced mentor echo of student work
- **Quality filter:** Kept rows where average rubric score >= 3.5 and minimum individual score >= 3 across accuracy, strategy, clarity, interpretation, and completeness
- **Split method:** Thread-level split (all examples from a given thread stay in the same split) with hash-based deduplication
- **Final splits:** 2,276 train / 138 valid / 141 test
- **Services covered:** Algebra, Geometry, Math Fundamentals, Pre-Algebra, and others

### Training

| Parameter | Value |
|-----------|-------|
| Base model | `meta-llama/Llama-3.1-8B-Instruct` |
| Method | QLoRA (4-bit NF4 quantization + LoRA r=16) |
| Trainable parameters | 41.9M / 8.07B (0.52%) |
| max_seq_length | 1024 |
| Epochs | 3 |
| Effective batch size | 8 (batch=1 x grad_accum=8) |
| GPU | Tesla T4 (16 GB VRAM) |
| Training time | ~4.5 hours |
| Final train loss | 1.06 |

### Key technical details

- **Chat template:** Uses `tokenizer.apply_chat_template` for proper Llama 3.1 formatting
- **Loss masking:** `DataCollatorForCompletionOnlyLM` ensures only assistant tokens contribute to loss
- **Smart truncation:** When conversations exceed `max_seq_length`, user content is trimmed from the tail before touching assistant content
- **Response template detection:** Probe-based approach to auto-detect assistant turn token IDs, avoiding tokenization boundary issues

### Results

See [`inference_results.md`](inference_results.md) for tracked model outputs across versions.

**v0.1 observations:**
- Model successfully learned MathForum mentor style (supportive tone, guiding questions, no direct answers)
- Known artifacts: hallucinated names from training data, echo of student work
- Planned fixes for v0.2: additional data cleaning (strip signatures, filter to first-response only)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./data/finetune.db` | Database connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `STORAGE_BACKEND` | `local` | `"local"` or `"minio"` |
| `STORAGE_PATH` | `./data` | Local file storage path |
| `HF_TOKEN` | — | HuggingFace token (required for gated models) |
| `HF_CACHE_DIR` | `./data/hf_cache` | Model cache directory |

## Data Pipeline (External)

A separate private pipeline converts raw mentoring threads into training-ready JSONL:

1. **Convert** thread files to structured JSON (system/user/assistant + metadata)
2. **Clean** content (remove images, base64, placeholders, encoding artifacts)
3. **Filter** by rubric quality scores (avg >= 3.5, min >= 3)
4. **Split** by thread ID into train/valid/test
5. **Deduplicate** across splits

The pipeline scripts and raw data are maintained separately and are not included in this repository.
