"""FastAPI backend for SLM Forge.

Provides REST endpoints consumed by the browser-based UI, and serves static UI
files (index.html, style.css, app.js).

All endpoints are prefixed with ``/api/`` and return JSON responses. The static
file mount is registered after all API routes to avoid route conflicts.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import (
    dataset_converter,
    exporter,
    hardware,
    job_manager,
    models_catalog,
    server_inference,
)

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
UI_DIR = BASE_DIR / "ui"
DATA_DIR = BASE_DIR / "jobs" / "datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SLM Forge API",
    description="REST API for the SLM Forge fine-tuning studio",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
@app.get("/api/hardware")
def api_hardware() -> dict[str, Any]:
    """Return detected hardware specs and the recommended training profile."""
    hw = hardware.detect_hardware()
    profile = hardware.recommend_profile(hw)
    return {"hardware": hw, "recommended_profile": profile}


# ---------------------------------------------------------------------------
# Models catalog
# ---------------------------------------------------------------------------
@app.get("/api/models")
def api_models() -> dict[str, Any]:
    """Return the list of supported SLMs, filtered by current hardware."""
    hw = hardware.detect_hardware()
    profile = hardware.recommend_profile(hw)
    return {
        "profile_key": profile["key"],
        "models": models_catalog.models_for_profile(profile),
    }


# ---------------------------------------------------------------------------
# Dataset upload
# ---------------------------------------------------------------------------
@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a JSON dataset file and return a format preview.

    The file is saved to ``jobs/datasets/`` and analyzed for format
    auto-detection. Accepted formats: Dataset Generator output, Alpaca,
    ShareGPT.
    """
    # Sanitize filename to prevent path traversal
    safe_name = Path(file.filename or "dataset.json").name
    dest = DATA_DIR / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        prev = dataset_converter.preview(str(dest), n=2)
    except Exception as e:
        raise HTTPException(400, f"Error reading dataset: {e}") from e

    return {"path": str(dest), **prev}


# ---------------------------------------------------------------------------
# Training lifecycle
# ---------------------------------------------------------------------------
@app.post("/api/train/start")
def api_train_start(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Start a new fine-tuning job.

    The request body must include ``dataset_path`` and all fields defined in
    ``trainer.TrainConfig``. The dataset is prepared (split into train/eval
    JSONL) before the training worker is launched as a detached subprocess.

    Returns the new job ID and dataset statistics.
    """
    dataset_path: str | None = payload.pop("dataset_path", None)
    if not dataset_path:
        raise HTTPException(400, "dataset_path is required.")

    # Prepare the dataset (convert format, split train/eval, write JSONL)
    job_id_placeholder = uuid.uuid4().hex[:8]
    prep_dir = BASE_DIR / "jobs" / "_prepared" / job_id_placeholder
    try:
        prep = dataset_converter.prepare_dataset(
            dataset_path, str(prep_dir)
        )
    except Exception as e:
        raise HTTPException(
            400, f"Error preparing dataset: {e}"
        ) from e

    # Build training configuration
    cfg: dict[str, Any] = {
        "base_model": payload.get("base_model"),
        "train_file": prep["train_file"],
        "eval_file": prep["eval_file"],
        "output_dir": payload.get(
            "output_dir", str(BASE_DIR / "jobs")
        ),
        "lora_r": payload.get("lora_r", 16),
        "lora_alpha": payload.get("lora_alpha", 32),
        "lora_dropout": payload.get("lora_dropout", 0.05),
        "learning_rate": payload.get("learning_rate", 2e-4),
        "num_train_epochs": payload.get("num_train_epochs", 3.0),
        "per_device_batch_size": payload.get("per_device_batch_size", 1),
        "grad_accum_steps": payload.get("grad_accum_steps", 8),
        "max_seq_len": payload.get("max_seq_len", 1024),
        "use_4bit": payload.get("use_4bit", False),
        "precision": payload.get("precision", "bf16"),
        "gradient_checkpointing": payload.get(
            "gradient_checkpointing", True
        ),
        "seed": payload.get("seed", 42),
        "resume_from_checkpoint": None,
        "hf_token": payload.get("hf_token"),
    }

    if not cfg["base_model"]:
        raise HTTPException(400, "base_model is required.")

    # Validate numeric fields to prevent configuration errors
    if cfg["num_train_epochs"] <= 0:
        raise HTTPException(400, "num_train_epochs must be positive.")
    if cfg["lora_r"] <= 0:
        raise HTTPException(400, "lora_r must be positive.")

    job_id = job_manager.start_job(cfg)
    return {
        "job_id": job_id,
        "dataset_stats": {
            "n_train": prep["n_train"],
            "n_eval": prep["n_eval"],
            "format": prep["format_detected"],
        },
    }


@app.post("/api/train/{job_id}/stop")
def api_train_stop(job_id: str) -> dict[str, Any]:
    """Stop a running training job."""
    return job_manager.stop_job(job_id)


@app.post("/api/train/{job_id}/resume")
def api_train_resume(job_id: str) -> dict[str, Any]:
    """Resume a stopped job from its last checkpoint."""
    new_id = job_manager.resume_job(job_id)
    return {"job_id": new_id}


# ---------------------------------------------------------------------------
# Job listing and status
# ---------------------------------------------------------------------------
@app.get("/api/jobs")
def api_jobs() -> list[dict[str, Any]]:
    """List all training jobs, newest first."""
    return job_manager.list_jobs()


@app.get("/api/jobs/{job_id}/status")
def api_job_status(job_id: str) -> dict[str, Any]:
    """Get the live status of a single job."""
    return job_manager.get_status(job_id)


# ---------------------------------------------------------------------------
# Export / merge
# ---------------------------------------------------------------------------
@app.post("/api/export/merge")
def api_export_merge(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Merge a trained LoRA adapter with its base model."""
    try:
        result = exporter.merge_lora(
            base_model=payload["base_model"],
            adapter_dir=payload["adapter_dir"],
            output_dir=payload["output_dir"],
            hf_token=payload.get("hf_token"),
        )
    except Exception as e:
        raise HTTPException(400, f"Error merging model: {e}") from e
    return result


@app.get("/api/export/gguf-instructions")
def api_gguf_instructions() -> dict[str, str]:
    """Return step-by-step instructions for GGUF export."""
    return {"instructions": exporter.GGUF_INSTRUCTIONS}


# ---------------------------------------------------------------------------
# Model serving (lightweight, in-process inference)
# ---------------------------------------------------------------------------
@app.post("/api/serve/start")
def api_serve_start(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Load a model into memory for quick testing."""
    try:
        server_inference.load_model(payload["model_dir"])
    except Exception as e:
        raise HTTPException(400, f"Error loading model: {e}") from e
    return {"ok": True, "loaded": True}


@app.post("/api/serve/stop")
def api_serve_stop() -> dict[str, Any]:
    """Unload the currently loaded model."""
    server_inference.unload_model()
    return {"ok": True}


@app.get("/api/serve/health")
def api_serve_health() -> dict[str, bool]:
    """Check whether a model is currently loaded."""
    return {"loaded": server_inference.is_loaded()}


@app.post("/api/serve/generate")
def api_serve_generate(
    payload: dict[str, Any] = Body(...),
) -> dict[str, str]:
    """Generate text from the loaded model."""
    try:
        text = server_inference.generate(
            prompt=payload["prompt"],
            max_new_tokens=payload.get("max_new_tokens", 256),
            temperature=payload.get("temperature", 0.7),
        )
    except Exception as e:
        raise HTTPException(400, str(e)) from e
    return {"text": text}


# ---------------------------------------------------------------------------
# Static UI files — MUST be mounted after all /api routes
# ---------------------------------------------------------------------------
app.mount(
    "/", StaticFiles(directory=str(UI_DIR), html=True), name="ui"
)
