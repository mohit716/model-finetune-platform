"""
Fine-tuning worker — runs LoRA training via HuggingFace Transformers + PEFT.

Supports two dataset formats:
  - "instruction": {"instruction", "input?", "output"}
  - "chat": {"messages": [{role, content}, ...]}  (OpenAI-style)

For chat format, loss is masked so only assistant tokens contribute.
"""

import os
import json
import time
import traceback
from datetime import datetime, timezone
from typing import Tuple
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
        ds_format = config.get("dataset_format", "auto")
        if ds_format == "auto":
            ds_format = detect_dataset_format(dataset_path)
        logger.info(f"Dataset format: {ds_format}")

        training_data, dataset_stats = load_and_validate_dataset(
            dataset_path, fmt=ds_format, config=config,
        )
        logger.info(f"Dataset loaded: {len(training_data)} samples")

        update_job(job_id, metrics={"dataset_stats": dataset_stats})

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

        # Enable gradient checkpointing to slash activation memory (~50% savings)
        if config.get("gradient_checkpointing", True) and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info("Gradient checkpointing enabled")

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

        # ── 6. TRAIN ─────────────────────────────────────
        logger.info(f"Starting training: {num_epochs} epochs, {total_steps} steps")

        progress_cb = TrainingProgressCallback(job_id, total_steps)

        from transformers import TrainerCallback

        class ProgressReporter(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and state:
                    progress_cb.on_log(logs, state.global_step, state.epoch or 0)

        if ds_format == "chat":
            # ── Chat path: apply_chat_template + assistant-only loss ──
            from trl import DataCollatorForCompletionOnlyLM

            _tok = tokenizer  # closure

            def chat_formatting_func(examples):
                texts = []
                for msgs in examples["messages"]:
                    msgs = _fit_messages_to_length(msgs, _tok, max_seq_length)
                    texts.append(
                        _tok.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=False
                        )
                    )
                return texts

            resp_ids = _build_response_template_ids(tokenizer)
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=resp_ids,
                tokenizer=tokenizer,
            )

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=training_data,
                tokenizer=tokenizer,
                formatting_func=chat_formatting_func,
                data_collator=data_collator,
                max_seq_length=max_seq_length,
                callbacks=[ProgressReporter()],
            )
        else:
            # ── Instruction path (original) ──────────────
            def instruction_formatting_func(examples):
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

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=training_data,
                tokenizer=tokenizer,
                formatting_func=instruction_formatting_func,
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
            "dataset_format": ds_format,
            "dataset_stats": dataset_stats,
        }

        # Calculate adapter size
        adapter_size = sum(
            os.path.getsize(os.path.join(adapter_dir, f))
            for f in os.listdir(adapter_dir)
            if os.path.isfile(os.path.join(adapter_dir, f))
        )

        # ── 8. REGISTER MODEL ────────────────────────────
        logger.info("Registering fine-tuned model...")
        model_config = dict(config)
        model_config["dataset_format"] = ds_format

        model_record = FineTunedModel(
            name=f"{job.name}_model",
            base_model=job.base_model,
            job_id=job_id,
            status=ModelStatus.READY,
            adapter_path=adapter_storage_key,
            metrics=metrics,
            config=model_config,
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


def detect_dataset_format(file_path: str) -> str:
    """Peek at the first valid line to decide 'chat' vs 'instruction'."""
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "messages" in record:
                return "chat"
            if "instruction" in record:
                return "instruction"
            raise ValueError(
                "Unrecognised format: first row needs 'messages' or 'instruction'"
            )
    raise ValueError("Dataset file is empty")


def load_and_validate_dataset(file_path: str, fmt: str = "instruction",
                              config: dict | None = None) -> Tuple:
    """
    Load JSONL and return (HFDataset, stats_dict).

    Dispatches to instruction or chat loader based on *fmt*.
    """
    if fmt == "chat":
        return _load_chat_dataset(file_path, config or {})
    return _load_instruction_dataset(file_path)


# ── Instruction format ───────────────────────────────────

def _load_instruction_dataset(file_path: str) -> Tuple:
    from datasets import Dataset as HFDataset

    records, dropped = [], 0
    output_lens = []

    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            if "instruction" not in record or "output" not in record:
                dropped += 1
                continue

            records.append({
                "instruction": record["instruction"],
                "input": record.get("input", ""),
                "output": record["output"],
            })
            output_lens.append(len(record["output"]))

    if not records:
        raise ValueError("Dataset must contain at least 1 valid sample")

    stats = {
        "format": "instruction",
        "total_rows": len(records) + dropped,
        "valid_rows": len(records),
        "dropped_rows": dropped,
        "avg_assistant_chars": round(sum(output_lens) / len(output_lens)),
        "max_assistant_chars": max(output_lens),
    }
    logger.info(f"Instruction dataset: {stats}")
    return HFDataset.from_list(records), stats


# ── Chat format ──────────────────────────────────────────

_REQUIRED_ROLES = ("system", "user", "assistant")


def _load_chat_dataset(file_path: str, config: dict) -> Tuple:
    """
    Load OpenAI-style chat JSONL.

    Validates:
      - Exactly 3 messages with roles [system, user, assistant]
      - Non-empty assistant content
      - Optional char-length filters from config
    Returns (HFDataset, stats).
    """
    from datasets import Dataset as HFDataset

    max_user = config.get("max_user_chars")
    max_asst = config.get("max_assistant_chars")

    records, dropped = [], 0
    user_lens, asst_lens = [], []

    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue

            msgs = record.get("messages")
            if not isinstance(msgs, list) or len(msgs) != 3:
                dropped += 1
                continue

            roles = tuple(m.get("role") for m in msgs)
            if roles != _REQUIRED_ROLES:
                dropped += 1
                continue

            sys_c = (msgs[0].get("content") or "").strip()
            usr_c = (msgs[1].get("content") or "").strip()
            ast_c = (msgs[2].get("content") or "").strip()

            if not ast_c:
                dropped += 1
                continue

            if max_user and len(usr_c) > max_user:
                dropped += 1
                continue
            if max_asst and len(ast_c) > max_asst:
                dropped += 1
                continue

            records.append({
                "messages": [
                    {"role": "system", "content": sys_c},
                    {"role": "user", "content": usr_c},
                    {"role": "assistant", "content": ast_c},
                ]
            })
            user_lens.append(len(usr_c))
            asst_lens.append(len(ast_c))

    if not records:
        raise ValueError(
            f"No valid chat samples (checked {dropped + len(records)} lines, "
            f"dropped {dropped})"
        )

    stats = {
        "format": "chat",
        "total_rows": len(records) + dropped,
        "valid_rows": len(records),
        "dropped_rows": dropped,
        "avg_user_chars": round(sum(user_lens) / len(user_lens)),
        "max_user_chars": max(user_lens),
        "avg_assistant_chars": round(sum(asst_lens) / len(asst_lens)),
        "max_assistant_chars": max(asst_lens),
    }
    logger.info(f"Chat dataset: {stats}")
    return HFDataset.from_list(records), stats


def _fit_messages_to_length(messages: list, tokenizer, max_seq_length: int) -> list:
    """
    If the tokenised conversation exceeds *max_seq_length*, trim the **user**
    message from the tail first.  Only touches assistant as a last resort.
    """
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    n_tok = len(tokenizer.encode(full, add_special_tokens=False))
    if n_tok <= max_seq_length:
        return messages

    excess = n_tok - max_seq_length
    trim_chars = int(excess * 4) + 40  # conservative char estimate

    usr = messages[1]["content"]
    if len(usr) - trim_chars >= 80:
        return [
            messages[0],
            {"role": "user", "content": usr[: len(usr) - trim_chars].rstrip() + " ..."},
            messages[2],
        ]

    # User too short to absorb all excess — keep 80 chars, trim assistant tail
    remaining = trim_chars - (len(usr) - 80)
    ast = messages[2]["content"]
    return [
        messages[0],
        {"role": "user", "content": usr[:80].rstrip() + " ..."},
        {"role": "assistant", "content": ast[: max(60, len(ast) - remaining)].rstrip()},
    ]


def _build_response_template_ids(tokenizer) -> list[int]:
    """
    Auto-detect the token-ID sequence that marks the beginning of the
    assistant turn in this tokenizer's chat template.

    Works by tokenizing two probe conversations with different assistant
    content, finding where the token IDs diverge, and extracting the
    assistant-header tokens just before the divergence point.
    This avoids tokenization boundary issues with standalone encoding.
    """
    probe_a = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "AAAA"},
    ]
    probe_b = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "BBBB"},
    ]

    text_a = tokenizer.apply_chat_template(
        probe_a, tokenize=False, add_generation_prompt=False
    )
    text_b = tokenizer.apply_chat_template(
        probe_b, tokenize=False, add_generation_prompt=False
    )
    ids_a = tokenizer.encode(text_a, add_special_tokens=False)
    ids_b = tokenizer.encode(text_b, add_special_tokens=False)

    diverge = 0
    for i in range(min(len(ids_a), len(ids_b))):
        if ids_a[i] != ids_b[i]:
            diverge = i
            break

    if diverge < 2:
        raise ValueError(
            "Could not detect assistant response template — "
            "tokenizer may not have a chat_template."
        )

    n_ctx = min(4, diverge)
    ids = ids_a[diverge - n_ctx: diverge]

    logger.info(
        f"Response template ({n_ctx} tokens): {ids} → "
        f"{repr(tokenizer.decode(ids))}"
    )
    return ids


def _detect_target_modules(model) -> list:
    """Auto-detect linear layer names for LoRA targeting."""
    import torch.nn as nn

    target_names = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear,)):
            short_name = name.split(".")[-1]
            if short_name not in ("lm_head",):
                target_names.add(short_name)

    targets = list(target_names)
    logger.info(f"Auto-detected LoRA target modules: {targets}")
    return targets if targets else ["q_proj", "v_proj"]
