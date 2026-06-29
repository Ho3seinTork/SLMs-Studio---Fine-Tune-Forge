"""Dataset format converter for SLM fine-tuning.

Converts datasets from multiple source formats (Dataset Generator output,
Alpaca, ShareGPT) into the standardized JSONL format required by
``trl.SFTTrainer`` with ``apply_chat_template``.

Supported input formats
-----------------------
- **Dataset Generator**: JSON array of objects with ``status``, ``conversation``,
  ``summary``, etc. Only entries with ``status == "success"`` are used.
- **Alpaca**: JSON array of objects with ``instruction``, ``input`` (optional),
  ``output`` fields.
- **ShareGPT**: JSON array of objects with ``conversations`` array containing
  ``from``/``value`` pairs.

Output format
-------------
JSONL with one JSON object per line::

    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Tuple


def _from_dataset_generator(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Dataset Generator output format to standard messages format."""
    out: list[dict[str, Any]] = []
    for sample in raw:
        if sample.get("status") != "success":
            continue
        conv = sample.get("conversation") or []
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in conv
            if m.get("content")
        ]
        if len(messages) >= 2:
            out.append({"messages": messages})
    return out


def _from_alpaca(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Alpaca format to standard messages format."""
    out: list[dict[str, Any]] = []
    for item in raw:
        if "instruction" not in item:
            continue
        user_content = item["instruction"]
        if item.get("input"):
            user_content += "\n\n" + item["input"]
        out.append(
            {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": item.get("output", "")},
                ]
            }
        )
    return out


def _from_sharegpt(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert ShareGPT format to standard messages format."""
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "system": "system",
        "user": "user",
        "assistant": "assistant",
    }
    out: list[dict[str, Any]] = []
    for item in raw:
        conv = item.get("conversations") or item.get("conversation")
        if not conv:
            continue
        messages = [
            {
                "role": role_map.get(
                    m.get("from") or m.get("role", ""), "user"
                ),
                "content": m.get("value") or m.get("content", ""),
            }
            for m in conv
        ]
        if len(messages) >= 2:
            out.append({"messages": messages})
    return out


def load_and_normalize(path: str) -> Tuple[list[dict[str, Any]], str]:
    """Load a dataset file and normalize it to the standard ``messages`` format.

    Args:
        path: Path to the JSON dataset file.

    Returns:
        Tuple of (normalized_examples, detected_format) where format is one of
        ``"dataset_generator"``, ``"alpaca"``, or ``"sharegpt"``.

    Raises:
        ValueError: If the file is empty, not a JSON array, or format is
            unrecognized.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(raw, list) or not raw:
        raise ValueError("Input file must be a non-empty JSON array.")

    first = raw[0]

    if "conversation" in first and "status" in first:
        examples = _from_dataset_generator(raw)
        fmt = "dataset_generator"
    elif "instruction" in first:
        examples = _from_alpaca(raw)
        fmt = "alpaca"
    elif "conversations" in first or (
        "conversation" in first and isinstance(first["conversation"], list)
    ):
        examples = _from_sharegpt(raw)
        fmt = "sharegpt"
    else:
        raise ValueError(
            "Unrecognized dataset format. "
            "Expected: Dataset Generator output, Alpaca, or ShareGPT."
        )

    if not examples:
        raise ValueError("No usable training examples found in the file.")

    return examples, fmt


def split_and_write(
    examples: list[dict[str, Any]],
    out_dir: str,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """Split a normalized dataset into train/eval sets and write as JSONL.

    Datasets with fewer than 10 examples skip the eval split entirely (all data
    goes to training).

    Args:
        examples: Normalized examples (list of ``{"messages": [...]}`` dicts).
        out_dir: Output directory path.
        eval_ratio: Fraction of data reserved for evaluation (default: 0.1).
        seed: Random seed for reproducible shuffling.

    Returns:
        dict with keys: ``train_file``, ``eval_file``, ``n_train``, ``n_eval``.
    """
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


def preview(path: str, n: int = 3) -> dict[str, Any]:
    """Load a dataset and return a preview without writing any files.

    Args:
        path: Path to the JSON dataset file.
        n: Number of sample examples to include.

    Returns:
        dict with ``format_detected``, ``total_examples``, and ``sample`` keys.
    """
    examples, fmt = load_and_normalize(path)
    return {
        "format_detected": fmt,
        "total_examples": len(examples),
        "sample": examples[:n],
    }


def prepare_dataset(
    input_path: str,
    out_dir: str,
    eval_ratio: float = 0.1,
) -> dict[str, Any]:
    """Full pipeline: load, normalize, split, and write a dataset.

    Args:
        input_path: Path to the input JSON dataset file.
        out_dir: Directory to write ``train.jsonl`` and ``eval.jsonl``.
        eval_ratio: Fraction of data for evaluation (default: 0.1).

    Returns:
        dict with ``train_file``, ``eval_file``, ``n_train``, ``n_eval``,
        ``format_detected``, and ``total_examples``.
    """
    examples, fmt = load_and_normalize(input_path)
    result = split_and_write(examples, out_dir, eval_ratio=eval_ratio)
    result["format_detected"] = fmt
    result["total_examples"] = len(examples)
    return result


if __name__ == "__main__":
    import sys

    print(json.dumps(preview(sys.argv[1]), indent=2, ensure_ascii=False))
