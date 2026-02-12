"""
Inference router — load fine-tuned models and generate text.

Prototype uses HuggingFace transformers directly.
Production should swap to vLLM or TGI for better throughput.
"""

import time
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from loguru import logger

from app.database import get_db
from app.models import FineTunedModel, ModelStatus
from app.schemas import InferenceRequest, InferenceResponse
from app.config import settings
from app.services.storage import storage

router = APIRouter(prefix="/api/v1/inference", tags=["Inference"])

# Simple in-memory model cache (production: use a proper model server)
_model_cache: dict = {}


def _load_model(model_record: FineTunedModel):
    """Load a fine-tuned model (base + LoRA adapter) into memory."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    cache_key = model_record.id

    if cache_key in _model_cache:
        logger.info(f"Model cache hit: {cache_key}")
        return _model_cache[cache_key]

    adapter_path = storage.get_file_path(model_record.adapter_path)
    logger.info(f"Loading model: {model_record.base_model} + adapter {model_record.adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        cache_dir=settings.HF_CACHE_DIR,
        token=settings.HF_TOKEN,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_record.base_model,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        cache_dir=settings.HF_CACHE_DIR,
        token=settings.HF_TOKEN,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    _model_cache[cache_key] = {"model": model, "tokenizer": tokenizer}
    logger.info(f"Model loaded and cached: {cache_key}")
    return _model_cache[cache_key]


@router.post("/", response_model=InferenceResponse)
def run_inference(req: InferenceRequest, db: Session = Depends(get_db)):
    """Run inference on a fine-tuned model."""
    import torch

    # Look up model
    model_record = db.query(FineTunedModel).filter(
        FineTunedModel.id == req.model_id
    ).first()

    if not model_record:
        raise HTTPException(404, f"Model {req.model_id} not found")
    if model_record.status != ModelStatus.READY:
        raise HTTPException(400, f"Model is not ready (status: {model_record.status})")

    # Load model
    try:
        loaded = _load_model(model_record)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(500, f"Failed to load model: {str(e)}")

    model = loaded["model"]
    tokenizer = loaded["tokenizer"]

    # Format prompt
    if req.system_prompt:
        formatted_prompt = (
            f"### System:\n{req.system_prompt}\n\n"
            f"### Instruction:\n{req.prompt}\n\n"
            f"### Response:\n"
        )
    else:
        formatted_prompt = (
            f"### Instruction:\n{req.prompt}\n\n"
            f"### Response:\n"
        )

    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=settings.MAX_SEQ_LENGTH,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=max(req.temperature, 0.01),
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    latency_ms = (time.time() - start_time) * 1000

    # Decode only the new tokens
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    logger.info(
        f"Inference: model={req.model_id} | "
        f"tokens={len(generated_ids)} | "
        f"latency={latency_ms:.0f}ms"
    )

    return InferenceResponse(
        model_id=req.model_id,
        prompt=req.prompt,
        generated_text=generated_text,
        tokens_generated=len(generated_ids),
        latency_ms=round(latency_ms, 2),
    )


@router.delete("/cache")
def clear_model_cache():
    """Clear the in-memory model cache to free GPU/RAM."""
    import torch

    _model_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model cache cleared")
    return {"status": "cache_cleared"}
