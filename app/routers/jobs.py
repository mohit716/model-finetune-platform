"""
Jobs router — CRUD + submission for fine-tuning jobs.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from loguru import logger

from app.database import get_db
from app.models import FineTuneJob, Dataset, JobStatus
from app.schemas import JobCreate, JobResponse, JobList

router = APIRouter(prefix="/api/v1/jobs", tags=["Jobs"])


@router.post("/", response_model=JobResponse, status_code=201)
def create_job(payload: JobCreate, db: Session = Depends(get_db)):
    """Submit a new fine-tuning job."""

    # Validate dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    if not dataset:
        raise HTTPException(404, f"Dataset {payload.dataset_id} not found")

    # Create job record
    job = FineTuneJob(
        name=payload.name,
        base_model=payload.base_model,
        dataset_id=payload.dataset_id,
        config=payload.config.model_dump(),
        total_epochs=payload.config.num_epochs,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Dispatch to Celery worker
    from app.workers.trainer import run_finetuning

    task = run_finetuning.delay(job.id)
    job.celery_task_id = task.id
    db.commit()

    logger.info(f"Job submitted: {job.id} → Celery task {task.id}")
    return job


@router.get("/", response_model=JobList)
def list_jobs(
    skip: int = 0,
    limit: int = 20,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List all fine-tuning jobs with optional status filter."""
    query = db.query(FineTuneJob)
    if status:
        try:
            status_enum = JobStatus(status)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}. Valid: {[s.value for s in JobStatus]}")
        query = query.filter(FineTuneJob.status == status_enum)

    total = query.count()
    jobs = (
        query
        .order_by(FineTuneJob.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return JobList(jobs=jobs, total=total)


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, db: Session = Depends(get_db)):
    """Get detailed job status and metrics."""
    job = db.query(FineTuneJob).filter(FineTuneJob.id == job_id).first()
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return job


@router.delete("/{job_id}")
def cancel_job(job_id: str, db: Session = Depends(get_db)):
    """Cancel a running or pending job."""
    job = db.query(FineTuneJob).filter(FineTuneJob.id == job_id).first()
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        raise HTTPException(400, f"Job is already {job.status}")

    # Revoke Celery task
    if job.celery_task_id:
        from app.workers.celery_app import celery_app
        celery_app.control.revoke(job.celery_task_id, terminate=True)

    job.status = JobStatus.CANCELLED
    db.commit()
    logger.info(f"Job cancelled: {job_id}")
    return {"status": "cancelled", "job_id": job_id}
