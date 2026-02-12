#!/usr/bin/env python3
"""
End-to-end test script — walks through the full fine-tuning workflow.

Usage:
    python scripts/test_e2e.py

Prerequisites:
    - API server running on localhost:8000
    - Redis running
    - Celery worker running
"""

import time
import sys
import httpx

BASE_URL = "http://localhost:8000"
client = httpx.Client(base_url=BASE_URL, timeout=30.0)


def log(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def main():
    # ── 1. Health Check ──────────────────────────────────
    log("1. Health Check")
    r = client.get("/health")
    print(r.json())
    assert r.status_code == 200

    # ── 2. Upload Dataset ────────────────────────────────
    log("2. Upload Dataset")
    with open("scripts/sample_dataset.jsonl", "rb") as f:
        r = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("sample_dataset.jsonl", f, "application/jsonl")},
            data={"name": "customer-support-demo"},
        )
    print(r.json())
    assert r.status_code == 201
    dataset_id = r.json()["id"]
    print(f"  Dataset ID: {dataset_id}")

    # ── 3. Submit Fine-Tuning Job ────────────────────────
    log("3. Submit Fine-Tuning Job")
    r = client.post(
        "/api/v1/jobs/",
        json={
            "name": "customer-support-v1",
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "dataset_id": dataset_id,
            "config": {
                "num_epochs": 1,
                "learning_rate": 2e-4,
                "batch_size": 2,
                "max_seq_length": 256,
                "lora_r": 8,
                "lora_alpha": 16,
                "gradient_accumulation_steps": 2,
            },
        },
    )
    print(r.json())
    assert r.status_code == 201
    job_id = r.json()["id"]
    print(f"  Job ID: {job_id}")

    # ── 4. Poll Job Status ───────────────────────────────
    log("4. Waiting for Training to Complete...")
    max_wait = 600  # 10 minutes
    start = time.time()

    while time.time() - start < max_wait:
        r = client.get(f"/api/v1/jobs/{job_id}")
        job = r.json()
        status = job["status"]
        loss = job.get("train_loss", "N/A")
        step = job.get("current_step", 0)
        total = job.get("total_steps", "?")

        print(f"  Status: {status} | Step: {step}/{total} | Loss: {loss}")

        if status == "completed":
            print("\n  Training completed!")
            model_id = job["output_model_id"]
            print(f"  Model ID: {model_id}")
            break
        elif status == "failed":
            print(f"\n  Training failed: {job.get('error_message')}")
            sys.exit(1)

        time.sleep(10)
    else:
        print("  Timeout waiting for training!")
        sys.exit(1)

    # ── 5. List Models ───────────────────────────────────
    log("5. List Fine-Tuned Models")
    r = client.get("/api/v1/models/")
    print(r.json())

    # ── 6. Run Inference ─────────────────────────────────
    log("6. Run Inference")
    r = client.post(
        "/api/v1/inference/",
        json={
            "model_id": model_id,
            "prompt": "How do I reset my password?",
            "max_new_tokens": 128,
            "temperature": 0.7,
        },
    )
    result = r.json()
    print(f"  Prompt:    {result.get('prompt')}")
    print(f"  Response:  {result.get('generated_text')}")
    print(f"  Tokens:    {result.get('tokens_generated')}")
    print(f"  Latency:   {result.get('latency_ms')}ms")

    log("ALL TESTS PASSED!")


if __name__ == "__main__":
    main()
