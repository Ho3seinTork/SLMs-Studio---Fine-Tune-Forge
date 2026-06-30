# -*- coding: utf-8 -*-
"""کاتالوگ مدل‌های زبانی کوچک (SLM) پشتیبانی‌شده | Curated SLM catalog."""

MODELS = [
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct", "label": "Qwen2.5 0.5B Instruct",
        "params_b": 0.5, "gated": False, "min_ram_gb_cpu": 4, "min_vram_gb_qlora": 2,
        "notes": "سریع‌ترین گزینه برای تست روی لپ‌تاپ ضعیف یا CPU."
    },
    {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "label": "TinyLlama 1.1B Chat",
        "params_b": 1.1, "gated": False, "min_ram_gb_cpu": 6, "min_vram_gb_qlora": 3,
        "notes": "گزینه پایدار و سبک برای آموزش روی CPU یا GPU کوچک."
    },
    {
        "id": "Qwen/Qwen2.5-1.5B-Instruct", "label": "Qwen2.5 1.5B Instruct",
        "params_b": 1.5, "gated": False, "min_ram_gb_cpu": 8, "min_vram_gb_qlora": 4,
        "notes": "کیفیت بهتر از 0.5B؛ روی ۸ گیگ رم با صبر زیاد روی CPU هم قابل اجراست."
    },
    {
        "id": "google/gemma-2-2b-it", "label": "Gemma 2 — 2B Instruct",
        "params_b": 2.0, "gated": True, "min_ram_gb_cpu": 10, "min_vram_gb_qlora": 5,
        "notes": "مدل گیتد است؛ نیاز به پذیرش لایسنس و توکن Hugging Face دارد (huggingface-cli login)."
    },
    {
        "id": "microsoft/Phi-3.5-mini-instruct", "label": "Phi-3.5 mini Instruct (~3.8B)",
        "params_b": 3.8, "gated": False, "min_ram_gb_cpu": 16, "min_vram_gb_qlora": 6,
        "notes": "کیفیت بالا؛ روی لپ‌تاپ ۸ گیگ فقط با GPU و QLoRA توصیه می‌شود، نه CPU."
    },
    {
        "id": "Qwen/Qwen2.5-3B-Instruct", "label": "Qwen2.5 3B Instruct",
        "params_b": 3.0, "gated": False, "min_ram_gb_cpu": 14, "min_vram_gb_qlora": 6,
        "notes": "برای سرور یا لپ‌تاپ با GPU مناسب‌تر است."
    },
    {
        "id": "meta-llama/Llama-3.2-3B-Instruct", "label": "Llama 3.2 — 3B Instruct",
        "params_b": 3.0, "gated": True, "min_ram_gb_cpu": 14, "min_vram_gb_qlora": 6,
        "notes": "گیتد است؛ نیاز به پذیرش لایسنس متا و توکن HF دارد."
    },
    {
        "id": "google/gemma-2-9b-it", "label": "Gemma 2 — 9B Instruct (فقط سرور)",
        "params_b": 9.0, "gated": True, "min_ram_gb_cpu": 32, "min_vram_gb_qlora": 12,
        "notes": "مناسب سرور با GPU قوی (≥۱۶ گیگ VRAM)؛ روی لپ‌تاپ ۸ گیگ رم عملاً قابل آموزش نیست."
    },
]


def models_for_profile(profile: dict) -> list:
    """فقط مدل‌هایی را برمی‌گرداند که با پروفایل سخت‌افزاری فعلی واقع‌بینانه جا می‌شوند."""
    max_b = profile.get("max_model_params_b", 2)
    out = []
    for m in MODELS:
        fits = m["params_b"] <= max_b
        out.append({**m, "fits_current_hardware": fits})
    return out
