# -*- coding: utf-8 -*-
"""
هسته آموزش LoRA/QLoRA | Core LoRA/QLoRA fine-tuning engine
---------------------------------------------------------------------------
این ماژول روی transformers + peft + trl + datasets ساخته شده است.
به‌صورت تدریجی import می‌کند تا اگر این کتابخانه‌ها نصب نباشند، فقط همین
ماژول هنگام واقعاً اجرای آموزش خطا بدهد، نه در زمان import کلی برنامه.

نکته صداقت فنی: روی CPU بدون GPU، آموزش حتی مدل‌های کوچک می‌تواند ساعت‌ها طول
بکشد. این ماژول 4bit (bitsandbytes) را فقط زمانی فعال می‌کند که CUDA موجود باشد.
"""
from __future__ import annotations
import json
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Callable


@dataclass
class TrainConfig:
    base_model: str
    train_file: str
    eval_file: Optional[str]
    output_dir: str
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    per_device_batch_size: int = 1
    grad_accum_steps: int = 8
    max_seq_len: int = 1024
    use_4bit: bool = False
    precision: str = "bf16"  # "fp32" | "fp16" | "bf16"
    gradient_checkpointing: bool = True
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    hf_token: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(text: str) -> "TrainConfig":
        return TrainConfig(**json.loads(text))


class StatusWriterCallback:
    """کال‌بک trl/transformers برای نوشتن وضعیت زنده (loss، step، ETA) روی دیسک."""

    def __init__(self, job_dir: Path, total_steps_hint: Optional[int] = None):
        self.job_dir = job_dir
        self.status_file = job_dir / "status.json"
        self.loss_file = job_dir / "loss_history.jsonl"
        self.start_time = time.time()
        self.total_steps_hint = total_steps_hint

    def write_status(self, **kwargs):
        data = {"updated_at": time.time(), "elapsed_sec": round(time.time() - self.start_time, 1)}
        data.update(kwargs)
        self.status_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def log_loss(self, step: int, loss: float, epoch: float):
        with self.loss_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "loss": loss, "epoch": epoch, "t": time.time()}, ensure_ascii=False) + "\n")


def _build_callback_class(status_writer: StatusWriterCallback):
    from transformers import TrainerCallback

    class _Callback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            status_writer.write_status(phase="training", status="running", step=0,
                                        total_steps=state.max_steps)

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" in logs:
                status_writer.log_loss(state.global_step, logs["loss"], state.epoch or 0)
            status_writer.write_status(
                phase="training", status="running",
                step=state.global_step, total_steps=state.max_steps,
                epoch=round(state.epoch or 0, 3),
                last_loss=logs.get("loss"), learning_rate=logs.get("learning_rate"),
            )

        def on_save(self, args, state, control, **kwargs):
            status_writer.write_status(phase="checkpoint_saved", status="running",
                                        step=state.global_step, total_steps=state.max_steps)

    return _Callback()


def run_training(cfg: TrainConfig, job_dir: str, stop_flag_path: Optional[str] = None):
    """
    اجرای واقعی آموزش. این تابع باید در یک پروسه مجزا (نه ترد UI) صدا زده شود
    چون مدل‌های زبانی، GIL و حافظه را به‌شدت اشغال می‌کنند — به همین دلیل
    job_manager.py آن را از طریق train_worker.py در یک subprocess کاملاً جدا اجرا می‌کند.
    """
    job_path = Path(job_dir)
    job_path.mkdir(parents=True, exist_ok=True)
    status_writer = StatusWriterCallback(job_path)
    status_writer.write_status(phase="loading_model", status="running", step=0)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset
    except ImportError as e:
        status_writer.write_status(phase="failed", status="failed",
                                    error=f"کتابخانه لازم نصب نیست: {e}. requirements.txt را نصب کنید.")
        raise

    dtype = torch.bfloat16 if cfg.precision == "bf16" else (torch.float16 if cfg.precision == "fp16" else torch.float32)
    quant_config = None
    if cfg.use_4bit and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, token=cfg.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"token": cfg.hf_token}
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = dtype
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32  # CPU پایدارتر با fp32 است

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    status_writer.write_status(phase="loading_dataset", status="running")
    data_files = {"train": cfg.train_file}
    if cfg.eval_file:
        data_files["validation"] = cfg.eval_file
    dataset = load_dataset("json", data_files=data_files)

    def format_example(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}

    dataset = dataset.map(format_example)

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

    callback = _build_callback_class(status_writer)
    trainer = SFTTrainer(
        model=model, args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        callbacks=[callback],
    )

    status_writer.write_status(phase="training", status="running", step=0)
    try:
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    except KeyboardInterrupt:
        status_writer.write_status(phase="stopped", status="stopped")
        return

    adapter_dir = job_path / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    status_writer.write_status(phase="done", status="done", adapter_dir=str(adapter_dir))
