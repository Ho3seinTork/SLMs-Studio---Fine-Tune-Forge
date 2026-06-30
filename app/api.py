# -*- coding: utf-8 -*-
"""
بک‌اند FastAPI | FastAPI backend
---------------------------------------------------------------------------
تمام endpoint های REST مورد نیاز UI را فراهم می‌کند و فایل‌های استاتیک UI
(ui/index.html, style.css, app.js) را نیز سرو می‌کند.
"""
from __future__ import annotations
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app import hardware, models_catalog, dataset_converter, job_manager, exporter, server_inference

BASE_DIR = Path(__file__).resolve().parent.parent
UI_DIR = BASE_DIR / "ui"
DATA_DIR = BASE_DIR / "jobs" / "datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SLM Forge API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/hardware")
def api_hardware():
    hw = hardware.detect_hardware()
    profile = hardware.recommend_profile(hw)
    return {"hardware": hw, "recommended_profile": profile}


@app.get("/api/models")
def api_models():
    hw = hardware.detect_hardware()
    profile = hardware.recommend_profile(hw)
    return {"profile_key": profile["key"], "models": models_catalog.models_for_profile(profile)}


@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    dest = DATA_DIR / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        prev = dataset_converter.preview(str(dest), n=2)
    except Exception as e:
        raise HTTPException(400, f"خطا در خواندن دیتاست: {e}")
    return {"path": str(dest), **prev}


@app.post("/api/train/start")
def api_train_start(payload: dict = Body(...)):
    """
    payload باید شامل dataset_path و فیلدهای TrainConfig باشد.
    ابتدا دیتاست را به train/eval jsonl تبدیل می‌کند، سپس job را استارت می‌کند.
    """
    dataset_path = payload.pop("dataset_path", None)
    if not dataset_path:
        raise HTTPException(400, "dataset_path الزامی است.")

    job_id_placeholder = uuid.uuid4().hex[:8]
    prep_dir = BASE_DIR / "jobs" / "_prepared" / job_id_placeholder
    try:
        prep = dataset_converter.prepare_dataset(dataset_path, str(prep_dir))
    except Exception as e:
        raise HTTPException(400, f"خطا در آماده‌سازی دیتاست: {e}")

    cfg = {
        "base_model": payload.get("base_model"),
        "train_file": prep["train_file"],
        "eval_file": prep["eval_file"],
        "output_dir": payload.get("output_dir", str(BASE_DIR / "jobs")),
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
        "gradient_checkpointing": payload.get("gradient_checkpointing", True),
        "seed": payload.get("seed", 42),
        "resume_from_checkpoint": None,
        "hf_token": payload.get("hf_token"),
    }
    if not cfg["base_model"]:
        raise HTTPException(400, "base_model الزامی است.")

    job_id = job_manager.start_job(cfg)
    return {"job_id": job_id, "dataset_stats": {"n_train": prep["n_train"], "n_eval": prep["n_eval"], "format": prep["format_detected"]}}


@app.post("/api/train/{job_id}/stop")
def api_train_stop(job_id: str):
    return job_manager.stop_job(job_id)


@app.post("/api/train/{job_id}/resume")
def api_train_resume(job_id: str):
    new_id = job_manager.resume_job(job_id)
    return {"job_id": new_id}


@app.get("/api/jobs")
def api_jobs():
    return job_manager.list_jobs()


@app.get("/api/jobs/{job_id}/status")
def api_job_status(job_id: str):
    return job_manager.get_status(job_id)


@app.post("/api/export/merge")
def api_export_merge(payload: dict = Body(...)):
    try:
        result = exporter.merge_lora(
            base_model=payload["base_model"], adapter_dir=payload["adapter_dir"],
            output_dir=payload["output_dir"], hf_token=payload.get("hf_token"),
        )
    except Exception as e:
        raise HTTPException(400, f"خطا در ادغام مدل: {e}")
    return result


@app.get("/api/export/gguf-instructions")
def api_gguf_instructions():
    return {"instructions": exporter.GGUF_INSTRUCTIONS}


@app.post("/api/serve/start")
def api_serve_start(payload: dict = Body(...)):
    try:
        server_inference.load_model(payload["model_dir"])
    except Exception as e:
        raise HTTPException(400, f"خطا در بارگذاری مدل: {e}")
    return {"ok": True, "loaded": True}


@app.post("/api/serve/stop")
def api_serve_stop():
    server_inference.unload_model()
    return {"ok": True}


@app.get("/api/serve/health")
def api_serve_health():
    return {"loaded": server_inference.is_loaded()}


@app.post("/api/serve/generate")
def api_serve_generate(payload: dict = Body(...)):
    try:
        text = server_inference.generate(
            prompt=payload["prompt"],
            max_new_tokens=payload.get("max_new_tokens", 256),
            temperature=payload.get("temperature", 0.7),
        )
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"text": text}


# فایل‌های استاتیک رابط کاربری | Static UI files (باید بعد از تمام route های /api تعریف شود)
app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
