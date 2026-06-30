# -*- coding: utf-8 -*-
"""
مدیر کارها | Job manager
---------------------------------------------------------------------------
این ماژول قلب «بدون قطع‌شدن روی سرور/Remote Desktop» است: هر آموزش به‌عنوان
یک پروسه سیستمی کاملاً مجزا (subprocess) اجرا می‌شود که از پروسه GUI/سرور API
جدا شده است. حتی اگر برنامه اصلی یا اتصال ریموت قطع شود، این پروسه روی دیسک
(status.json / loss_history.jsonl / log.txt) پیشرفت می‌نویسد و با باز کردن
دوباره برنامه، وضعیتش از روی همین فایل‌ها بازیابی می‌شود.
"""
from __future__ import annotations
import json
import os
import platform
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

JOBS_ROOT = Path(__file__).resolve().parent.parent / "jobs"
JOBS_ROOT.mkdir(parents=True, exist_ok=True)


def _new_job_id() -> str:
    return uuid.uuid4().hex[:10]


def start_job(train_config_dict: dict) -> str:
    job_id = _new_job_id()
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    (job_dir / "config.json").write_text(json.dumps(train_config_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = job_dir / "log.txt"
    (job_dir / "status.json").write_text(json.dumps({"phase": "queued", "status": "running"}), encoding="utf-8")

    cmd = [sys.executable, "-m", "app.train_worker", "--job-dir", str(job_dir)]
    log_file = open(log_path, "ab", buffering=0)

    popen_kwargs = dict(
        stdout=log_file, stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    if platform.system() == "Windows":
        # کاملاً از کنسول/نشست فعلی جدا می‌شود تا با قطع RDP یا بسته‌شدن ترمینال نمیرد
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        # روی لینوکس/مک: یک session جدید می‌سازد (مثل nohup) تا با قطع SSH نمیرد
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    (job_dir / "pid.txt").write_text(str(proc.pid), encoding="utf-8")
    return job_id


def _pid_alive(pid: int) -> bool:
    if psutil is not None:
        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_status(job_id: str) -> dict:
    job_dir = JOBS_ROOT / job_id
    if not job_dir.exists():
        return {"error": "job_not_found"}

    status_path = job_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}

    pid_path = job_dir / "pid.txt"
    pid = int(pid_path.read_text().strip()) if pid_path.exists() else None
    alive = _pid_alive(pid) if pid else False

    if status.get("status") == "running" and pid is not None and not alive:
        status["status"] = "crashed_or_stopped"

    loss_history = []
    loss_path = job_dir / "loss_history.jsonl"
    if loss_path.exists():
        lines = loss_path.read_text(encoding="utf-8").strip().splitlines()
        loss_history = [json.loads(l) for l in lines[-200:]]

    log_tail = ""
    log_path = job_dir / "log.txt"
    if log_path.exists():
        try:
            log_tail = log_path.read_text(encoding="utf-8", errors="ignore")[-4000:]
        except Exception:
            log_tail = ""

    status["job_id"] = job_id
    status["pid"] = pid
    status["pid_alive"] = alive
    status["loss_history"] = loss_history
    status["log_tail"] = log_tail
    return status


def stop_job(job_id: str) -> dict:
    job_dir = JOBS_ROOT / job_id
    pid_path = job_dir / "pid.txt"
    if not pid_path.exists():
        return {"error": "pid_not_found"}
    pid = int(pid_path.read_text().strip())
    try:
        if psutil is not None and psutil.pid_exists(pid):
            p = psutil.Process(pid)
            p.terminate()
            try:
                p.wait(timeout=8)
            except psutil.TimeoutExpired:
                p.kill()
        elif platform.system() != "Windows":
            os.kill(pid, signal.SIGTERM)
    except Exception as e:
        return {"error": str(e)}

    status_path = job_dir / "status.json"
    data = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    data.update({"phase": "stopped", "status": "stopped"})
    status_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True}


def resume_job(job_id: str) -> str:
    """یک job متوقف‌شده را با resume_from_checkpoint روی آخرین چک‌پوینت دوباره اجرا می‌کند."""
    old_job_dir = JOBS_ROOT / job_id
    cfg = json.loads((old_job_dir / "config.json").read_text(encoding="utf-8"))
    ckpt_root = old_job_dir / "checkpoints"
    last_ckpt = None
    if ckpt_root.exists():
        checkpoints = sorted(ckpt_root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if checkpoints:
            last_ckpt = str(checkpoints[-1])
    cfg["resume_from_checkpoint"] = last_ckpt
    return start_job(cfg)


def list_jobs() -> list:
    jobs = []
    for d in sorted(JOBS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir() and (d / "config.json").exists():
            jobs.append(get_status(d.name))
    return jobs
