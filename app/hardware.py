# -*- coding: utf-8 -*-
"""
تشخیص سخت‌افزار و پیشنهاد پروفایل آموزش | Hardware detection & training profile advisor
---------------------------------------------------------------------------
این ماژول مشخصات سیستم (RAM، CPU، GPU/VRAM) را تشخیص می‌دهد و بر اساس آن
یک پروفایل پیشنهادی (لپ‌تاپ ضعیف / لپ‌تاپ با GPU کوچک / سرور قدرتمند) برمی‌گرداند.
طراحی شده تا حتی بدون نصب‌بودن torch هم بدون خطا اجرا شود (GPU detection به‌صورت
اختیاری و امن انجام می‌شود).
"""
from __future__ import annotations
import platform
import os

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def _bytes_to_gb(n: int) -> float:
    return round(n / (1024 ** 3), 2)


def detect_hardware() -> dict:
    """مشخصات فعلی سیستم را برمی‌گرداند | Returns current system specs."""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu_name": platform.processor() or "نامشخص",
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
        import torch  # noqa: WPS433  (import اختیاری و امن)
        info["torch_installed"] = True
        if torch.cuda.is_available():
            info["gpu_available"] = True
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_vram_total_gb"] = _bytes_to_gb(props.total_memory)
            info["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception:
        # torch نصب نیست یا خطایی در GPU detection رخ داد — بدون مشکل ادامه بده
        pass

    return info


# ---------------------------------------------------------------------------
# پروفایل‌های پیشنهادی آموزش | Suggested training profiles
# ---------------------------------------------------------------------------
PROFILES = {
    "laptop_cpu": {
        "label": "لپ‌تاپ بدون GPU اختصاصی (CPU فقط)",
        "max_model_params_b": 1.6,
        "use_4bit": False,
        "precision": "fp32",
        "lora_r": 8,
        "lora_alpha": 16,
        "per_device_batch_size": 1,
        "grad_accum_steps": 16,
        "max_seq_len": 512,
        "gradient_checkpointing": True,
        "warning": "آموزش روی CPU بسیار کندتر از GPU است (احتمالاً چند ساعت تا یک شب برای هر epoch روی دیتاست کوچک). "
                   "فقط مدل‌های کوچک (≤۱.۵ میلیارد پارامتر) و دیتاست کوچک پیشنهاد می‌شود."
    },
    "laptop_gpu_small": {
        "label": "لپ‌تاپ با GPU کوچک (مثلاً ۴ تا ۶ گیگ VRAM)",
        "max_model_params_b": 4,
        "use_4bit": True,
        "precision": "bf16",
        "lora_r": 16,
        "lora_alpha": 32,
        "per_device_batch_size": 1,
        "grad_accum_steps": 8,
        "max_seq_len": 1024,
        "gradient_checkpointing": True,
        "warning": "با QLoRA (۴-بیتی) مدل‌های تا حدود ۴ میلیارد پارامتر معمولاً روی این سخت‌افزار جا می‌شوند."
    },
    "server_gpu": {
        "label": "سرور با GPU قدرتمند (≥ ۱۶ گیگ VRAM)",
        "max_model_params_b": 14,
        "use_4bit": True,
        "precision": "bf16",
        "lora_r": 32,
        "lora_alpha": 64,
        "per_device_batch_size": 4,
        "grad_accum_steps": 2,
        "max_seq_len": 2048,
        "gradient_checkpointing": False,
        "warning": "می‌توانید 4bit را خاموش کنید و LoRA معمولی با دقت bf16 اجرا کنید تا کیفیت بالاتر برود."
    },
}


def recommend_profile(hw: dict | None = None) -> dict:
    """بر اساس مشخصات سخت‌افزار، یک پروفایل پیشنهادی انتخاب می‌کند."""
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
    print(json.dumps({"hardware": hw_info, "recommended_profile": rec}, indent=2, ensure_ascii=False))
