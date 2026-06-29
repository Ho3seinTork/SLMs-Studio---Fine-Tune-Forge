"""Curated catalog of supported Small Language Models (SLMs).

Each entry specifies the model identifier on Hugging Face Hub, parameter count,
gating requirements, and minimum hardware requirements for both CPU-only and
QLoRA-based fine-tuning.
"""

from __future__ import annotations

from typing import Any

MODELS: list[dict[str, Any]] = [
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Qwen2.5 0.5B Instruct",
        "params_b": 0.5,
        "gated": False,
        "min_ram_gb_cpu": 4,
        "min_vram_gb_qlora": 2,
        "notes": "Fastest option for testing on low-resource laptops or CPU-only systems.",
    },
    {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "label": "TinyLlama 1.1B Chat",
        "params_b": 1.1,
        "gated": False,
        "min_ram_gb_cpu": 6,
        "min_vram_gb_qlora": 3,
        "notes": "Stable, lightweight choice for CPU or small GPU training.",
    },
    {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "label": "Qwen2.5 1.5B Instruct",
        "params_b": 1.5,
        "gated": False,
        "min_ram_gb_cpu": 8,
        "min_vram_gb_qlora": 4,
        "notes": "Better quality than 0.5B; runs on 8 GB RAM with patience on CPU.",
    },
    {
        "id": "google/gemma-2-2b-it",
        "label": "Gemma 2 — 2B Instruct",
        "params_b": 2.0,
        "gated": True,
        "min_ram_gb_cpu": 10,
        "min_vram_gb_qlora": 5,
        "notes": (
            "Gated model; requires accepting the license on Hugging Face and "
            "providing an access token."
        ),
    },
    {
        "id": "microsoft/Phi-3.5-mini-instruct",
        "label": "Phi-3.5 mini Instruct (~3.8B)",
        "params_b": 3.8,
        "gated": False,
        "min_ram_gb_cpu": 16,
        "min_vram_gb_qlora": 6,
        "notes": "High quality; on 8 GB laptops only viable with GPU + QLoRA, not CPU.",
    },
    {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "label": "Qwen2.5 3B Instruct",
        "params_b": 3.0,
        "gated": False,
        "min_ram_gb_cpu": 14,
        "min_vram_gb_qlora": 6,
        "notes": "Best suited for servers or laptops with a dedicated GPU.",
    },
    {
        "id": "meta-llama/Llama-3.2-3B-Instruct",
        "label": "Llama 3.2 — 3B Instruct",
        "params_b": 3.0,
        "gated": True,
        "min_ram_gb_cpu": 14,
        "min_vram_gb_qlora": 6,
        "notes": "Gated model; requires accepting Meta's license terms on Hugging Face.",
    },
    {
        "id": "google/gemma-2-9b-it",
        "label": "Gemma 2 — 9B Instruct (server only)",
        "params_b": 9.0,
        "gated": True,
        "min_ram_gb_cpu": 32,
        "min_vram_gb_qlora": 12,
        "notes": (
            "For powerful GPU servers (≥16 GB VRAM) only. "
            "Not practical on consumer laptops."
        ),
    },
]


def models_for_profile(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Return models compatible with the given hardware profile.

    Args:
        profile: Training profile dict from hardware.recommend_profile().

    Returns:
        List of model entries, each augmented with a ``fits_current_hardware``
        boolean indicating whether the model can realistically run on the
        current hardware.
    """
    max_b = float(profile.get("max_model_params_b", 2))
    result: list[dict[str, Any]] = []
    for m in MODELS:
        fits = float(m["params_b"]) <= max_b
        result.append({**m, "fits_current_hardware": fits})
    return result
