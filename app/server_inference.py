# -*- coding: utf-8 -*-
"""
سرور استنتاج سبک | Lightweight inference server
---------------------------------------------------------------------------
مدل فاین‌تیون‌شده (یا ادغام‌شده) را یک‌بار در حافظه بارگذاری می‌کند و از طریق
یک endpoint ساده در دسترس می‌گذارد — یعنی «مدل روی سرور سوار می‌شود».
"""
from __future__ import annotations
import threading

_LOCK = threading.Lock()
_PIPELINE = None
_LOADED_MODEL_DIR = None


def load_model(model_dir: str):
    global _PIPELINE, _LOADED_MODEL_DIR
    with _LOCK:
        if _LOADED_MODEL_DIR == model_dir and _PIPELINE is not None:
            return
        import torch
        from transformers import pipeline
        _PIPELINE = pipeline(
            "text-generation", model=model_dir, device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        _LOADED_MODEL_DIR = model_dir


def is_loaded() -> bool:
    return _PIPELINE is not None


def unload_model():
    global _PIPELINE, _LOADED_MODEL_DIR
    with _LOCK:
        _PIPELINE = None
        _LOADED_MODEL_DIR = None


def generate(prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    if _PIPELINE is None:
        raise RuntimeError("هیچ مدلی بارگذاری نشده است. ابتدا /api/serve/start را صدا بزنید.")
    out = _PIPELINE(
        prompt, max_new_tokens=max_new_tokens, temperature=temperature,
        do_sample=temperature > 0, pad_token_id=_PIPELINE.tokenizer.eos_token_id,
    )
    text = out[0]["generated_text"]
    return text[len(prompt):] if text.startswith(prompt) else text
