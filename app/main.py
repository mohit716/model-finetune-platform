"""
LLM Fine-Tuning Platform — Main FastAPI Application

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings
from app.database import init_db

# ── Routers ──────────────────────────────────────────────
from app.routers import jobs, models, datasets, inference


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Self-hosted LLM fine-tuning platform. "
        "Upload datasets, submit LoRA fine-tuning jobs, "
        "and run inference on your custom models."
    ),
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus Metrics (optional) ────────────────────────
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    logger.info("Prometheus metrics enabled at /metrics")
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not installed, metrics disabled")

# ── Register Routers ─────────────────────────────────────
app.include_router(jobs.router)
app.include_router(models.router)
app.include_router(datasets.router)
app.include_router(inference.router)


# ── Health Check ─────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """Basic health check endpoint."""
    import torch

    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    gpu_memory = None
    if gpu_available:
        mem = torch.cuda.get_device_properties(0).total_mem
        gpu_memory = f"{mem / 1e9:.1f} GB"

    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_memory": gpu_memory,
        "storage_backend": settings.STORAGE_BACKEND,
        "inference_backend": settings.INFERENCE_BACKEND,
    }


@app.get("/", tags=["System"])
def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
