"""Model export and LoRA merge utilities.

After fine-tuning, the trained LoRA adapter can be merged with the base model
to produce a standalone model directory compatible with any inference server
(transformers, llama.cpp, Ollama, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def merge_lora(
    base_model: str,
    adapter_dir: str,
    output_dir: str,
    hf_token: Optional[str] = None,
) -> dict[str, str]:
    """Merge a LoRA adapter with its base model and save the combined model.

    The merge-and-unload operation fuses the adapter weights into the base
    model, producing a standard transformers-compatible model directory.

    Args:
        base_model: Hugging Face model ID or path to the base model.
        adapter_dir: Path to the trained LoRA adapter (contains
            ``adapter_config.json`` and ``adapter_model.safetensors``).
        output_dir: Directory where the merged model will be saved.
        hf_token: Optional Hugging Face access token for gated models.

    Returns:
        dict with ``output_dir`` key pointing to the merged model path.

    Note:
        This function loads the full base model into memory and is therefore
        resource-intensive. Ensure sufficient RAM/VRAM is available.
    """
    import torch  # noqa: WPS433
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433
    from peft import PeftModel  # noqa: WPS433

    compute_dtype = (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=compute_dtype,
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
To export your fine-tuned model to GGUF format (for high-speed CPU inference
with llama.cpp or Ollama), follow these steps:

1. Clone and install llama.cpp:

   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp && pip install -r requirements.txt

2. Convert the merged model to GGUF:

   python convert_hf_to_gguf.py <your_merged_model_dir> \\
       --outfile model.gguf --outtype q4_k_m

3. Run directly with llama.cpp or import into Ollama:

   ollama create my-model -f Modelfile

   Where Modelfile contains: FROM ./model.gguf
"""
