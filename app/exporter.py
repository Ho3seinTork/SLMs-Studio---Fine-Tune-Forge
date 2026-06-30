# -*- coding: utf-8 -*-
"""
ادغام و خروجی‌گیری مدل | LoRA merge & export
---------------------------------------------------------------------------
آداپتور LoRA آموزش‌دیده را با مدل پایه ادغام می‌کند تا یک مدل کامل و مستقل
(قابل سرو با هر سروری مثل transformers، llama.cpp یا Ollama) بسازد.
"""
from __future__ import annotations
from pathlib import Path


def merge_lora(base_model: str, adapter_dir: str, output_dir: str, hf_token: str | None = None) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=hf_token,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    merged = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    return {"output_dir": str(out_path)}


GGUF_INSTRUCTIONS = """
برای خروجی‌گیری GGUF (جهت اجرا با llama.cpp یا Ollama روی CPU با سرعت بالا)،
این پروژه به‌صورت خودکار GGUF نمی‌سازد (نیاز به build کردن llama.cpp دارد) اما
مراحل دستی ساده است:

۱) کلون llama.cpp:
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp && pip install -r requirements.txt

۲) تبدیل مدل ادغام‌شده به GGUF:
   python convert_hf_to_gguf.py <output_dir شما> --outfile model.gguf --outtype q4_k_m

۳) اجرا با llama.cpp یا وارد کردن به Ollama:
   ollama create my-model -f Modelfile   (با اشاره به model.gguf در Modelfile)
"""
