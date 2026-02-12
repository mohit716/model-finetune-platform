"""
Datasets router — upload and manage training datasets.
"""

import os
import json
import uuid
import tempfile
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from loguru import logger

from app.database import get_db
from app.models import Dataset
from app.schemas import DatasetResponse, DatasetList
from app.services.storage import storage
from app.config import settings

router = APIRouter(prefix="/api/v1/datasets", tags=["Datasets"])


@router.post("/upload", response_model=DatasetResponse, status_code=201)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    Upload a training dataset (JSONL format).

    Each line should be a JSON object with:
    - "instruction" (required): The task instruction
    - "input" (optional): Additional context
    - "output" (required): The expected response
    """
    # Validate file extension
    if not file.filename.endswith((".jsonl", ".json")):
        raise HTTPException(400, "Only .jsonl and .json files are supported")

    # Read and validate content
    content = await file.read()
    lines = content.decode("utf-8").strip().split("\n")
    num_samples = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            raise HTTPException(400, f"Invalid JSON on line {i + 1}")

        if "instruction" not in record:
            raise HTTPException(400, f"Line {i + 1}: missing 'instruction' field")
        if "output" not in record:
            raise HTTPException(400, f"Line {i + 1}: missing 'output' field")
        num_samples += 1

    if num_samples == 0:
        raise HTTPException(400, "Dataset is empty")

    # Save file
    dataset_id = str(uuid.uuid4())
    file_key = f"datasets/{dataset_id}/{file.filename}"

    # Write to temp then store (cross-platform temp dir)
    fd, tmp_path = tempfile.mkstemp(suffix=file.filename)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        storage.save_file(tmp_path, file_key)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # Create DB record
    dataset = Dataset(
        id=dataset_id,
        name=name,
        file_path=file_key,
        file_size=len(content),
        num_samples=num_samples,
        format="jsonl",
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    logger.info(f"Dataset uploaded: {name} ({num_samples} samples, {len(content)} bytes)")
    return dataset


@router.get("/", response_model=DatasetList)
def list_datasets(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    """List all uploaded datasets."""
    total = db.query(Dataset).count()
    datasets = (
        db.query(Dataset)
        .order_by(Dataset.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return DatasetList(datasets=datasets, total=total)


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Get dataset details."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    return dataset


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Delete a dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(404, f"Dataset {dataset_id} not found")

    storage.delete_file(dataset.file_path)
    db.delete(dataset)
    db.commit()
    logger.info(f"Deleted dataset: {dataset_id}")
    return {"status": "deleted", "dataset_id": dataset_id}
