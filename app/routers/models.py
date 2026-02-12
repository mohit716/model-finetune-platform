"""
Models router — list, inspect, and delete fine-tuned models.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from loguru import logger

from app.database import get_db
from app.models import FineTunedModel
from app.schemas import ModelResponse, ModelList
from app.services.storage import storage

router = APIRouter(prefix="/api/v1/models", tags=["Models"])


@router.get("/", response_model=ModelList)
def list_models(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    """List all fine-tuned models."""
    total = db.query(FineTunedModel).count()
    models = (
        db.query(FineTunedModel)
        .order_by(FineTunedModel.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return ModelList(models=models, total=total)


@router.get("/{model_id}", response_model=ModelResponse)
def get_model(model_id: str, db: Session = Depends(get_db)):
    """Get model details including metrics and config."""
    model = db.query(FineTunedModel).filter(FineTunedModel.id == model_id).first()
    if not model:
        raise HTTPException(404, f"Model {model_id} not found")
    return model


@router.delete("/{model_id}")
def delete_model(model_id: str, db: Session = Depends(get_db)):
    """Delete a fine-tuned model and its artifacts."""
    model = db.query(FineTunedModel).filter(FineTunedModel.id == model_id).first()
    if not model:
        raise HTTPException(404, f"Model {model_id} not found")

    # Delete adapter files (adapter_path is a storage key)
    try:
        storage.delete_file(model.adapter_path)
    except Exception as e:
        logger.warning(f"Failed to delete model files: {e}")

    db.delete(model)
    db.commit()
    logger.info(f"Deleted model: {model_id}")
    return {"status": "deleted", "model_id": model_id}
