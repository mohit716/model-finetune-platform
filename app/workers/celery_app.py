"""
Celery application — task queue for async fine-tuning jobs.
"""

from celery import Celery
from app.config import settings

celery_app = Celery(
    "finetune_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,                 # Re-queue if worker dies
    worker_prefetch_multiplier=1,        # One job at a time per worker
    worker_max_tasks_per_child=1,        # Restart worker after each job (free GPU memory)
    task_soft_time_limit=86400,          # 24h soft limit
    task_time_limit=90000,               # 25h hard limit
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["app.workers"])
