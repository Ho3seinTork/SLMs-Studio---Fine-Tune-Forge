"""Hardware detection and training profile recommendation.

This module detects system specifications (RAM, CPU, GPU/VRAM) and recommends
an appropriate training profile based on available resources. It is designed to
operate gracefully even when optional dependencies (torch, psutil) are not installed.
"""

from __future__ import annotations

import os
import platform
from typing import Any, Optional

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def _bytes_to_gb(n: int) -> float:
    """Convert bytes to gigabytes with two-decimal precision."""
    return round(n / (1024**3), 2)


def detect_hardware() -> dict[str, Any]:
    """Detect current system hardware specifications.

    Returns a dictionary containing OS details, CPU information, RAM capacity,
    and GPU availability (with VRAM and CUDA version if applicable).

    Returns:
        dict: Hardware specifications including os, cpu_name, cpu_cores_logical,
            ram_total_gb, ram_available_gb, gpu_available, gpu_name,
            gpu_vram_total_gb, cuda_version, torch_installed.
    """
    info: dict[str, Any] = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu_name": platform.processor() or "Unknown",
        "cpu_cores_logical": os.cpu_count() or 1,
        "ram_total_gb": None,
        "ram_available_gb": None,
        "gpu_available": False,
        "gpu_name": None,
        "gpu_vram_total_gb": None,
        "cuda_version": None,
        "torch_installed": False,
    }

    if psutil is not None:
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = _bytes_to_gb(vm.total)
        info["ram_available_gb"] = _bytes_to_gb(vm.available)

    try:
        import torch  # noqa: WPS433 (optional, safe import)

        info["torch_installed"] = True
        if torch.cuda.is_available():
            info["gpu_available"] = True
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_vram_total_gb"] = _bytes_to_gb(props.total_memory)
            info["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception:
        pass  # torch not installed or GPU detection failed — continue gracefully

    return info


# ---------------------------------------------------------------------------
# Training profile definitions
# ---------------------------------------------------------------------------

PROFILES: dict[str, dict[str, Any]] = {
    "laptop_cpu": {
        "label": "Laptop without dedicated GPU (CPU only)",
        "max_model_params_b": 1.6,
        "use_4bit": False,
        "precision": "fp32",
        "lora_r": 8,
        "lora_alpha": 16,
        "per_device_batch_size": 1,
        "grad_accum_steps": 16,
        "max_seq_len": 512,
        "gradient_checkpointing": True,
        "warning": (
            "Training on CPU is significantly slower than on GPU "
            "(potentially several hours per epoch on small datasets). "
            "Only models ≤1.5B parameters with small datasets are recommended."
        ),
    },
    "laptop_gpu_small": {
        "label": "Laptop with small GPU (4–6 GB VRAM)",
        "max_model_params_b": 4,
        "use_4bit": True,
        "precision": "bf16",
        "lora_r": 16,
        "lora_alpha": 32,
        "per_device_batch_size": 1,
        "grad_accum_steps": 8,
        "max_seq_len": 1024,
        "gradient_checkpointing": True,
        "warning": (
            "With QLoRA (4-bit quantization), models up to ~4B parameters "
            "typically fit on this hardware tier."
        ),
    },
    "server_gpu": {
        "label": "Server with powerful GPU (≥16 GB VRAM)",
        "max_model_params_b": 14,
        "use_4bit": True,
        "precision": "bf16",
        "lora_r": 32,
        "lora_alpha": 64,
        "per_device_batch_size": 4,
        "grad_accum_steps": 2,
        "max_seq_len": 2048,
        "gradient_checkpointing": False,
        "warning": (
            "Consider disabling 4-bit quantization and using full bf16 precision "
            "LoRA for higher quality results on this hardware tier."
        ),
    },
}


def recommend_profile(hw: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Select a training profile based on detected hardware specifications.

    Args:
        hw: Hardware specs dict from detect_hardware(). If None, auto-detects.

    Returns:
        dict: Profile with key, label, recommended hyperparameters, and warnings.
    """
    hw = hw or detect_hardware()

    if hw.get("gpu_available"):
        vram = hw.get("gpu_vram_total_gb") or 0
        profile_key = "server_gpu" if vram >= 16 else "laptop_gpu_small"
    else:
        profile_key = "laptop_cpu"

    profile = dict(PROFILES[profile_key])
    profile["key"] = profile_key
    profile["hardware"] = hw
    return profile


if __name__ == "__main__":
    import json

    hw_info = detect_hardware()
    rec = recommend_profile(hw_info)
    print(
        json.dumps(
            {"hardware": hw_info, "recommended_profile": rec},
            indent=2,
            ensure_ascii=False,
        )
    )
