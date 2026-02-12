# 🔧 LLM Fine-Tuning Platform (Prototype)

An open-source replica of Amazon SageMaker's fine-tuning infrastructure.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  FastAPI     │────▶│  Redis +     │────▶│  Celery Worker  │
│  REST API    │     │  Celery      │     │  (HF + PEFT)    │
└─────┬───────┘     └──────────────┘     └────────┬────────┘
      │                                           │
      ▼                                           ▼
┌─────────────┐                          ┌─────────────────┐
│ PostgreSQL   │                          │  Storage (Local/ │
│ (metadata)   │                          │  MinIO S3)       │
└─────────────┘                          └────────┬────────┘
                                                  │
                                                  ▼
                                         ┌─────────────────┐
                                         │  vLLM / HF      │
                                         │  Inference       │
                                         └─────────────────┘
```

## Quick Start

### Option 1: Docker Compose (Recommended)
```bash
docker-compose up --build
```

### Option 2: Local Development
Use a virtual environment so dependencies don't install globally:

```bash
# Create and activate venv (keeps your system Python clean)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Redis
redis-server &

# Start Celery worker
celery -A app.workers.celery_app worker --loglevel=info &

# Start API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Jobs
- `POST   /api/v1/jobs/`              — Submit a fine-tuning job
- `GET    /api/v1/jobs/`              — List all jobs
- `GET    /api/v1/jobs/{job_id}`      — Get job status & details
- `DELETE /api/v1/jobs/{job_id}`      — Cancel a job

### Models
- `GET    /api/v1/models/`            — List fine-tuned models
- `GET    /api/v1/models/{model_id}`  — Get model details
- `DELETE /api/v1/models/{model_id}`  — Delete a model

### Inference
- `POST   /api/v1/inference/`         — Run inference on a fine-tuned model

### Datasets
- `POST   /api/v1/datasets/upload`    — Upload a training dataset
- `GET    /api/v1/datasets/`          — List datasets

### Health
- `GET    /health`                    — Health check

## Example Usage

### 1. Upload a dataset
```bash
curl -X POST http://localhost:8000/api/v1/datasets/upload \
  -F "file=@my_dataset.jsonl" \
  -F "name=customer-support-data"
```

### 2. Submit a fine-tuning job
```bash
curl -X POST http://localhost:8000/api/v1/jobs/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-first-finetune",
    "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "dataset_id": "YOUR_DATASET_ID",
    "config": {
      "num_epochs": 3,
      "learning_rate": 2e-4,
      "batch_size": 4,
      "max_seq_length": 512,
      "lora_r": 16,
      "lora_alpha": 32,
      "lora_dropout": 0.05
    }
  }'
```

### 3. Check job status
```bash
curl http://localhost:8000/api/v1/jobs/{job_id}
```

### 4. Run inference
```bash
curl -X POST http://localhost:8000/api/v1/inference/ \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "YOUR_MODEL_ID",
    "prompt": "How do I reset my password?",
    "max_new_tokens": 256,
    "temperature": 0.7
  }'
```

## Dataset Format

JSONL with `instruction`, `input` (optional), and `output` fields:
```jsonl
{"instruction": "Summarize the following", "input": "Long text here...", "output": "Summary here"}
{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}
```

## Configuration

See `app/config.py` for all configuration options. Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./data/finetune.db` | Database connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `STORAGE_BACKEND` | `local` | `local` or `minio` |
| `STORAGE_PATH` | `./data` | Local storage path |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `HF_TOKEN` | `` | HuggingFace token for gated models |
