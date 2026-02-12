"""
Fine-tuning worker — runs LoRA training via HuggingFace Transformers + PEFT.

This is the heart of the platform. It:
1. Validates the dataset
2. Downloads/loads the base model
3. Applies LoRA configuration
4. Trains with the SFTTrainer
5. Saves the adapter and registers the model
"""

import os
import json
import time
import traceback
from datetime import datetime, timezone
from loguru import logger

from app.workers.celery_app import celery_app
from app.config import settings
from app.database import SessionLocal
from app.models import FineTuneJob, FineTunedModel, Dataset, JobStatus, ModelStatus
from app.services.storage import storage


def update_job(job_id: str, **kwargs):
    """Helper to update job fields in the database."""
    db = SessionLocal()
    try:
        job = db.query(FineTuneJob).filter(FineTuneJob.id == job_id).first()
        if job:
            for key, value in kwargs.items():
                setattr(job, key, value)
            db.commit()
    finally:
        db.close()


class TrainingProgressCallback:
    """Custom callback to report training progress back to the database."""

    def __init__(self, job_id: str, total_steps: int):
        self.job_id = job_id
        self.total_steps = total_steps
        self.last_update = 0

    def on_log(self, logs: dict, step: int, epoch: float):
        now = time.time()
        # Throttle DB updates to every 5 seconds
        if now - self.last_update < 5:
            return
        self.last_update = now

        update_kwargs = {
            "current_step": step,
            "current_epoch": int(epoch),
        }
        if "loss" in logs:
            update_kwargs["train_loss"] = round(logs["loss"], 4)
        if "eval_loss" in logs:
            update_kwargs["eval_loss"] = round(logs["eval_loss"], 4)

        update_job(self.job_id, **update_kwargs)
        logger.info(f"[{self.job_id}] Step {step}/{self.total_steps} | {logs}")


@celery_app.task(bind=True, name="app.workers.trainer.run_finetuning")
def run_finetuning(self, job_id: str):
    """
    Main fine-tuning task.

    Executed asynchronously by Celery worker with GPU access.
    """
    logger.info(f"Starting fine-tuning job: {job_id}")

    db = SessionLocal()
    try:
        job = db.query(FineTuneJob).filter(FineTuneJob.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return {"status": "error", "message": "Job not found"}

        # ── 1. VALIDATE ──────────────────────────────────
        update_job(job_id, status=JobStatus.VALIDATING,
                   started_at=datetime.now(timezone.utc))

        dataset_record = db.query(Dataset).filter(
            Dataset.id == job.dataset_id
        ).first()
        if not dataset_record:
            raise ValueError(f"Dataset {job.dataset_id} not found")

        dataset_path = storage.get_file_path(dataset_record.file_path)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        config = job.config or {}
        logger.info(f"Config: {config}")

        # ── 2. LOAD DATASET ──────────────────────────────
        logger.info("Loading and validating dataset...")
        training_data = load_and_validate_dataset(dataset_path)
        logger.info(f"Dataset loaded: {len(training_data)} samples")

        # ── 3. DOWNLOAD MODEL ────────────────────────────
        update_job(job_id, status=JobStatus.DOWNLOADING)
        logger.info(f"Loading base model: {job.base_model}")

        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer

        # Quantization config for memory efficiency
        use_4bit = torch.cuda.is_available()
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            job.base_model,
            trust_remote_code=True,
            cache_dir=settings.HF_CACHE_DIR,
            token=settings.HF_TOKEN,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            job.base_model,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            cache_dir=settings.HF_CACHE_DIR,
            token=settings.HF_TOKEN,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        model.config.use_cache = False

        # ── 4. CONFIGURE LoRA ────────────────────────────
        update_job(job_id, status=JobStatus.TRAINING)
        logger.info("Applying LoRA configuration...")

        # Auto-detect target modules if not specified
        target_modules = config.get("lora_target_modules")
        if not target_modules:
            target_modules = _detect_target_modules(model)

        lora_config = LoraConfig(
            r=config.get("lora_r", settings.DEFAULT_LORA_R),
            lora_alpha=config.get("lora_alpha", settings.DEFAULT_LORA_ALPHA),
            lora_dropout=config.get("lora_dropout", settings.DEFAULT_LORA_DROPOUT),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )

        model = get_peft_model(model, lora_config)
        trainable_params, all_params = model.get_nb_trainable_parameters()
        logger.info(
            f"LoRA params: {trainable_params:,} trainable / "
            f"{all_params:,} total ({100 * trainable_params / all_params:.2f}%)"
        )

        # ── 5. SETUP TRAINING ────────────────────────────
        output_dir = os.path.join(
            settings.STORAGE_PATH, "models", f"job_{job_id}"
        )
        os.makedirs(output_dir, exist_ok=True)

        num_epochs = config.get("num_epochs", settings.DEFAULT_NUM_EPOCHS)
        batch_size = config.get("batch_size", settings.DEFAULT_BATCH_SIZE)
        grad_accum = config.get(
            "gradient_accumulation_steps", settings.GRADIENT_ACCUMULATION_STEPS
        )

        total_steps = (len(training_data) // (batch_size * grad_accum)) * num_epochs
        update_job(job_id, total_epochs=num_epochs, total_steps=max(total_steps, 1))

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=config.get("learning_rate", settings.DEFAULT_LEARNING_RATE),
            weight_decay=config.get("weight_decay", settings.DEFAULT_WEIGHT_DECAY),
            warmup_steps=config.get("warmup_steps", settings.DEFAULT_WARMUP_STEPS),
            logging_steps=10,
            save_strategy="epoch",
            fp16=config.get("fp16", True) and torch.cuda.is_available(),
            optim="adamw_torch",
            report_to="none",
            max_grad_norm=0.3,
            lr_scheduler_type="cosine",
            save_total_limit=2,
            dataloader_pin_memory=False,
        )

        # Format dataset for SFTTrainer
        max_seq_length = config.get("max_seq_length", settings.MAX_SEQ_LENGTH)

        def formatting_func(examples):
            """Format each example into the chat/instruct template."""
            texts = []
            for i in range(len(examples["instruction"])):
                instruction = examples["instruction"][i]
                inp = examples.get("input", [""] * len(examples["instruction"]))[i]
                output = examples["output"][i]

                if inp:
                    text = (
                        f"### Instruction:\n{instruction}\n\n"
                        f"### Input:\n{inp}\n\n"
                        f"### Response:\n{output}"
                    )
                else:
                    text = (
                        f"### Instruction:\n{instruction}\n\n"
                        f"### Response:\n{output}"
                    )
                texts.append(text)
            return texts

        # ── 6. TRAIN ─────────────────────────────────────
        logger.info(f"Starting training: {num_epochs} epochs, {total_steps} steps")

        progress_cb = TrainingProgressCallback(job_id, total_steps)

        from transformers import TrainerCallback

        class ProgressReporter(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and state:
                    progress_cb.on_log(logs, state.global_step, state.epoch or 0)

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=training_data,
            processing_class=tokenizer,
            formatting_func=formatting_func,
            max_seq_length=max_seq_length,
            callbacks=[ProgressReporter()],
        )

        train_result = trainer.train()

        # ── 7. SAVE ADAPTER ──────────────────────────────
        logger.info("Saving LoRA adapter...")
        adapter_dir = os.path.join(output_dir, "adapter")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        # Store storage key (not absolute path) so delete and inference work with storage backend
        adapter_storage_key = f"models/job_{job_id}/adapter"

        # Save training metrics
        metrics = {
            "train_loss": round(train_result.training_loss, 4),
            "train_runtime": round(train_result.metrics.get("train_runtime", 0), 2),
            "train_samples_per_second": round(
                train_result.metrics.get("train_samples_per_second", 0), 2
            ),
            "total_steps": train_result.metrics.get("train_steps", 0),
            "trainable_params": trainable_params,
            "total_params": all_params,
        }

        # Calculate adapter size
        adapter_size = sum(
            os.path.getsize(os.path.join(adapter_dir, f))
            for f in os.listdir(adapter_dir)
            if os.path.isfile(os.path.join(adapter_dir, f))
        )

        # ── 8. REGISTER MODEL ────────────────────────────
        logger.info("Registering fine-tuned model...")
        model_record = FineTunedModel(
            name=f"{job.name}_model",
            base_model=job.base_model,
            job_id=job_id,
            status=ModelStatus.READY,
            adapter_path=adapter_storage_key,
            metrics=metrics,
            config=config,
            size_bytes=adapter_size,
        )
        db.add(model_record)
        db.commit()
        db.refresh(model_record)

        # ── 9. COMPLETE ──────────────────────────────────
        update_job(
            job_id,
            status=JobStatus.COMPLETED,
            output_model_id=model_record.id,
            output_path=adapter_storage_key,
            train_loss=metrics.get("train_loss"),
            metrics=metrics,
            completed_at=datetime.now(timezone.utc),
            current_step=total_steps,
            current_epoch=num_epochs,
        )

        logger.info(
            f"Job {job_id} completed! Model: {model_record.id} | "
            f"Loss: {metrics['train_loss']}"
        )
        return {
            "status": "completed",
            "model_id": model_record.id,
            "metrics": metrics,
        }

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}\n{traceback.format_exc()}")
        update_job(
            job_id,
            status=JobStatus.FAILED,
            error_message=str(e),
            completed_at=datetime.now(timezone.utc),
        )
        return {"status": "failed", "error": str(e)}

    finally:
        db.close()
        # Free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ── Helper Functions ─────────────────────────────────────

def load_and_validate_dataset(file_path: str):
    """Load a JSONL dataset and validate its structure."""
    from datasets import Dataset as HFDataset

    records = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON on line {i + 1}")

            if "instruction" not in record or "output" not in record:
                raise ValueError(
                    f"Line {i + 1} missing required fields: 'instruction' and 'output'"
                )
            records.append({
                "instruction": record["instruction"],
                "input": record.get("input", ""),
                "output": record["output"],
            })

    if len(records) < 1:
        raise ValueError("Dataset must contain at least 1 sample")

    logger.info(f"Validated {len(records)} samples")
    return HFDataset.from_list(records)


def _detect_target_modules(model) -> list:
    """Auto-detect linear layer names for LoRA targeting."""
    import torch.nn as nn

    target_names = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear,)):
            # Get the last part of the name (e.g., "q_proj" from "model.layers.0.self_attn.q_proj")
            short_name = name.split(".")[-1]
            if short_name not in ("lm_head",):  # Skip output head
                target_names.add(short_name)

    targets = list(target_names)
    logger.info(f"Auto-detected LoRA target modules: {targets}")
    return targets if targets else ["q_proj", "v_proj"]  # Fallback
