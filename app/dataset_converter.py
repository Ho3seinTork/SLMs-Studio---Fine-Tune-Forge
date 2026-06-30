# -*- coding: utf-8 -*-
"""
مبدل دیتاست | Dataset converter
---------------------------------------------------------------------------
ورودی اصلی: همان dataset.json که اپ «تولیدکننده دیتاست» (HTML/JS) می‌سازد:
    [ { "status": "success", "conversation": [ {"role":"system|user|assistant","content":"..."} , ... ],
        "summary": "...", "patientProfile": {...}, "writerOutput": {...}, ... }, ... ]

همچنین به‌صورت پشتیبان از فرمت‌های Alpaca و ShareGPT هم استفاده می‌کند.
خروجی: فایل JSONL با ساختار {"messages":[{role,content}, ...]} که مستقیماً برای
trl.SFTTrainer / apply_chat_template قابل استفاده است.
"""
from __future__ import annotations
import json
import random
from pathlib import Path


def _from_dataset_generator(raw: list) -> list:
    out = []
    for sample in raw:
        if sample.get("status") != "success":
            continue
        conv = sample.get("conversation") or []
        messages = [{"role": m["role"], "content": m["content"]} for m in conv if m.get("content")]
        if len(messages) >= 2:
            out.append({"messages": messages})
    return out


def _from_alpaca(raw: list) -> list:
    out = []
    for item in raw:
        if "instruction" not in item:
            continue
        user_content = item["instruction"]
        if item.get("input"):
            user_content += "\n\n" + item["input"]
        out.append({"messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": item.get("output", "")}
        ]})
    return out


def _from_sharegpt(raw: list) -> list:
    role_map = {"human": "user", "gpt": "assistant", "system": "system", "user": "user", "assistant": "assistant"}
    out = []
    for item in raw:
        conv = item.get("conversations") or item.get("conversation")
        if not conv:
            continue
        messages = [{"role": role_map.get(m.get("from") or m.get("role"), "user"), "content": m.get("value") or m.get("content", "")} for m in conv]
        if len(messages) >= 2:
            out.append({"messages": messages})
    return out


def load_and_normalize(path: str) -> list:
    """فایل ورودی را خوانده و به فرمت یکسان {"messages":[...]} تبدیل می‌کند."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("فایل ورودی باید یک آرایه JSON غیرخالی باشد.")

    first = raw[0]
    if "conversation" in first and "status" in first:
        examples = _from_dataset_generator(raw)
        fmt = "dataset_generator"
    elif "instruction" in first:
        examples = _from_alpaca(raw)
        fmt = "alpaca"
    elif "conversations" in first or ("conversation" in first and isinstance(first["conversation"], list)):
        examples = _from_sharegpt(raw)
        fmt = "sharegpt"
    else:
        raise ValueError("فرمت دیتاست شناخته نشد (انتظار: خروجی Dataset Generator، Alpaca یا ShareGPT).")

    if not examples:
        raise ValueError("هیچ نمونه موفق/قابل‌استفاده‌ای در فایل پیدا نشد.")
    return examples, fmt


def split_and_write(examples: list, out_dir: str, eval_ratio: float = 0.1, seed: int = 42):
    """دیتاست را به train/eval تقسیم کرده و به صورت JSONL ذخیره می‌کند."""
    rnd = random.Random(seed)
    items = examples[:]
    rnd.shuffle(items)
    n_eval = max(1, int(len(items) * eval_ratio)) if len(items) >= 10 else 0
    eval_items = items[:n_eval]
    train_items = items[n_eval:]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    train_file = out_path / "train.jsonl"
    eval_file = out_path / "eval.jsonl"

    with train_file.open("w", encoding="utf-8") as f:
        for ex in train_items:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    if eval_items:
        with eval_file.open("w", encoding="utf-8") as f:
            for ex in eval_items:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return {
        "train_file": str(train_file),
        "eval_file": str(eval_file) if eval_items else None,
        "n_train": len(train_items),
        "n_eval": len(eval_items),
    }


def preview(path: str, n: int = 3) -> dict:
    examples, fmt = load_and_normalize(path)
    return {
        "format_detected": fmt,
        "total_examples": len(examples),
        "sample": examples[:n],
    }


def prepare_dataset(input_path: str, out_dir: str, eval_ratio: float = 0.1) -> dict:
    examples, fmt = load_and_normalize(input_path)
    result = split_and_write(examples, out_dir, eval_ratio=eval_ratio)
    result["format_detected"] = fmt
    result["total_examples"] = len(examples)
    return result


if __name__ == "__main__":
    import sys
    print(json.dumps(preview(sys.argv[1]), indent=2, ensure_ascii=False))
