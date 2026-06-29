"""LoRA/QLoRA fine-tuning engine for Small Language Models.

This module provides the core training logic built on top of the Hugging Face
ecosystem: ``transformers``, ``peft``, ``trl``, and ``datasets``. It supports:

- Full-precision LoRA (fp32, fp16, bf16)
- 4-bit quantized QLoRA (via ``bitsandbytes``, requires CUDA)
- Live status reporting via a ``StatusWriterCallback`` that persists progress
  to disk (status.json, loss_history.jsonl) for real-time monitoring by the UI
  or job manager.

Dependencies are imported lazily so that the module can be imported without
all ML libraries installed; import errors only surface when ``run_training()``
is actually called.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass
class TrainConfig:
    """Configuration for a LoRA/QLoRA fine-tuning run.

    Attributes:
        base_model: Hugging Face model ID (e.g. ``Qwen/Qwen2.5-1.5B-Instruct``).
        train_file: Path to the training JSONL file.
        eval_file: Optional path to the evaluation JSONL file.
        output_dir: Root directory for job outputs (checkpoints, adapter, logs).
        lora_r: LoRA rank (default: 16).
        lora_alpha: LoRA scaling factor (default: 32).
        lora_dropout: LoRA dropout probability (default: 0.05).
        learning_rate: Peak learning rate (default: 2e-4).
        num_train_epochs: Number of training epochs (default: 3.0).
        per_device_batch_size: Batch size per device (default: 1).
        grad_accum_steps: Gradient accumulation steps (default: 8).
        max_seq_len: Maximum sequence length for tokenization (default: 1024).
        use_4bit: Enable 4-bit quantization (QLoRA via bitsandbytes, CUDA-only).
        precision: Compute precision: ``"fp32"``, ``"fp16"``, or ``"bf16"``.
        gradient_checkpointing: Enable gradient checkpointing to reduce memory.
        seed: Random seed for reproducibility (default: 42).
        resume_from_checkpoint: Optional path to a checkpoint directory.
        hf_token: Optional Hugging Face access token for gated models.
    """

    base_model: str
    train_file: str
    eval_file: Optional[str] = None
    output_dir: str = ""
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    per_device_batch_size: int = 1
    grad_accum_steps: int = 8
    max_seq_len: int = 1024
    use_4bit: bool = False
    precision: str = "bf16"
    gradient_checkpointing: bool = True
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    hf_token: Optional[str] = None

    def to_json(self) -> str:
        """Serialize configuration to a JSON string."""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(text: str) -> "TrainConfig":
        """Deserialize configuration from a JSON string."""
        return TrainConfig(**json.loads(text))


class StatusWriterCallback:
    """Writes live training status and loss history to disk.

    Each job directory receives two files:

    - ``status.json``: Current phase, step progress, last loss, timestamps.
    - ``loss_history.jsonl``: Append-only log of (step, loss, epoch, time)
      tuples for charting.

    These files are read by the job manager and UI to provide real-time
    monitoring without requiring inter-process communication.
    """

    def __init__(
        self,
        job_dir: Path,
        total_steps_hint: Optional[int] = None,
    ) -> None:
        self.job_dir = job_dir
        self.status_file = job_dir / "status.json"
        self.loss_file = job_dir / "loss_history.jsonl"
        self.start_time = time.time()
        self.total_steps_hint = total_steps_hint

    def write_status(self, **kwargs: Any) -> None:
        """Write a status snapshot to ``status.json``."""
        data: dict[str, Any] = {
            "updated_at": time.time(),
            "elapsed_sec": round(time.time() - self.start_time, 1),
        }
        data.update(kwargs)
        self.status_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def log_loss(self, step: int, loss: float, epoch: float) -> None:
        """Append a loss data point to ``loss_history.jsonl``."""
        with self.loss_file.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"step": step, "loss": loss, "epoch": epoch, "t": time.time()},
                    ensure_ascii=False,
                )
                + "\n",
            )


def _build_callback(sw: StatusWriterCallback) -> Any:
    """Build a Hugging Face ``TrainerCallback`` that delegates to a
    ``StatusWriterCallback``."""
    from transformers import TrainerCallback  # noqa: WPS433

    class _Callback(TrainerCallback):
        def on_train_begin(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> None:
            sw.write_status(
                phase="training",
                status="running",
                step=0,
                total_steps=state.max_steps,
            )

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: Optional[dict[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            logs = logs or {}
            if "loss" in logs:
                sw.log_loss(
                    state.global_step,
                    logs["loss"],
                    state.epoch or 0,
                )
            sw.write_status(
                phase="training",
                status="running",
                step=state.global_step,
                total_steps=state.max_steps,
                epoch=round(state.epoch or 0, 3),
                last_loss=logs.get("loss"),
                learning_rate=logs.get("learning_rate"),
            )

        def on_save(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> None:
            sw.write_status(
                phase="checkpoint_saved",
                status="running",
                step=state.global_step,
                total_steps=state.max_steps,
            )

    return _Callback()


def run_training(
    cfg: TrainConfig,
    job_dir: str,
    stop_flag_path: Optional[str] = None,
) -> None:
    """Execute a full LoRA/QLoRA fine-tuning run.

    This function should be called from a dedicated OS process (not a thread),
    as language models heavily consume GPU memory and CPU resources. The job
    manager uses ``train_worker.py`` to invoke this in a detached subprocess.

    Progress is written to disk throughout training — see
    ``StatusWriterCallback`` for details.

    Args:
        cfg: Fully populated training configuration.
        job_dir: Directory for this job's outputs (checkpoints, adapter, logs).
        stop_flag_path: Reserved for future external stop-signal support.

    Raises:
        ImportError: If required ML libraries are not installed.
    """
    job_path = Path(job_dir)
    job_path.mkdir(parents=True, exist_ok=True)
    sw = StatusWriterCallback(job_path)
    sw.write_status(phase="loading_model", status="running", step=0)

    # ------------------------------------------------------------------
    # Lazy imports — only fail here, not at module import time
    # ------------------------------------------------------------------
    try:
        import torch  # noqa: WPS433
        from transformers import (  # noqa: WPS433
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from peft import (  # noqa: WPS433
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from trl import SFTTrainer, SFTConfig  # noqa: WPS433
        from datasets import load_dataset  # noqa: WPS433
    except ImportError as e:
        sw.write_status(
            phase="failed",
            status="failed",
            error=f"Required library not installed: {e}. "
            f"Install dependencies from requirements.txt.",
        )
        raise

    # ------------------------------------------------------------------
    # Determine dtype and quantization config
    # ------------------------------------------------------------------
    if cfg.precision == "bf16":
        dtype = torch.bfloat16
    elif cfg.precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    quant_config: Any = None
    if cfg.use_4bit and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    # ------------------------------------------------------------------
    # Load tokenizer and model
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, token=cfg.hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"token": cfg.hf_token}
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = dtype
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)

    if quant_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.gradient_checkpointing,
        )
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ------------------------------------------------------------------
    # Apply LoRA adapters
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # ------------------------------------------------------------------
    # Load and format dataset
    # ------------------------------------------------------------------
    sw.write_status(phase="loading_dataset", status="running")

    data_files: dict[str, str] = {"train": cfg.train_file}
    if cfg.eval_file:
        data_files["validation"] = cfg.eval_file
    dataset = load_dataset("json", data_files=data_files)

    def _format_chat(example: dict[str, Any]) -> dict[str, str]:
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    dataset = dataset.map(_format_chat)

    # ------------------------------------------------------------------
    # Configure and run the trainer
    # ------------------------------------------------------------------
    sft_config = SFTConfig(
        output_dir=str(job_path / "checkpoints"),
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if cfg.eval_file else "no",
        max_seq_length=cfg.max_seq_len,
        bf16=(cfg.precision == "bf16" and torch.cuda.is_available()),
        fp16=(cfg.precision == "fp16" and torch.cuda.is_available()),
        seed=cfg.seed,
        report_to=[],
        dataset_text_field="text",
    )

    callback = _build_callback(sw)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        callbacks=[callback],
    )

    sw.write_status(phase="training", status="running", step=0)
    try:
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    except KeyboardInterrupt:
        sw.write_status(phase="stopped", status="stopped")
        return

    # ------------------------------------------------------------------
    # Save the trained adapter
    # ------------------------------------------------------------------
    adapter_dir = job_path / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    sw.write_status(
        phase="done", status="done", adapter_dir=str(adapter_dir)
    )
