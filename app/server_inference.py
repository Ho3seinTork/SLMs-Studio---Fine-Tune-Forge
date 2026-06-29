"""Lightweight in-memory model serving for quick evaluation.

Loads a fine-tuned (or merged) model once into memory and exposes it through
a simple text-generation interface. Intended for interactive testing within
the SLM Forge desktop UI, not for production serving.

Thread-safe: all operations are guarded by a module-level lock.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

_LOCK = threading.Lock()
_PIPELINE: Any = None
_LOADED_MODEL_DIR: Optional[str] = None


def load_model(model_dir: str) -> None:
    """Load a model from disk into memory for inference.

    If the same model directory is already loaded, this is a no-op.

    Args:
        model_dir: Path to a transformers-compatible model directory.

    Note:
        This operation is memory-intensive. Ensure sufficient RAM/VRAM is
        available before calling.
    """
    global _PIPELINE, _LOADED_MODEL_DIR  # noqa: PLW0603
    with _LOCK:
        if _LOADED_MODEL_DIR == model_dir and _PIPELINE is not None:
            return

        import torch  # noqa: WPS433
        from transformers import pipeline  # noqa: WPS433

        _PIPELINE = pipeline(
            "text-generation",
            model=model_dir,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=(
                torch.float16
                if torch.cuda.is_available()
                else torch.float32
            ),
        )
        _LOADED_MODEL_DIR = model_dir


def is_loaded() -> bool:
    """Return True if a model is currently loaded in memory."""
    return _PIPELINE is not None


def unload_model() -> None:
    """Unload the currently loaded model and free memory."""
    global _PIPELINE, _LOADED_MODEL_DIR  # noqa: PLW0603
    with _LOCK:
        _PIPELINE = None
        _LOADED_MODEL_DIR = None


def generate(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate text from the currently loaded model.

    Args:
        prompt: Input text prompt.
        max_new_tokens: Maximum number of tokens to generate (default: 256).
        temperature: Sampling temperature; 0 means greedy decoding
            (default: 0.7).

    Returns:
        The generated text (excluding the input prompt).

    Raises:
        RuntimeError: If no model is loaded.
    """
    if _PIPELINE is None:
        raise RuntimeError(
            "No model is loaded. Call load_model() first."
        )

    out = _PIPELINE(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=_PIPELINE.tokenizer.eos_token_id,
    )
    text: str = out[0]["generated_text"]
    return text[len(prompt) :] if text.startswith(prompt) else text
