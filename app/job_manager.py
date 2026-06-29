"""Job lifecycle manager for training runs.

Each fine-tuning job is executed as a fully detached OS subprocess, ensuring
that training survives disconnections (SSH, Remote Desktop) and that the job
manager process can be restarted independently. Job state is persisted entirely
on disk under the ``jobs/`` directory.
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
from typing import Any, Optional

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

JOBS_ROOT = Path(__file__).resolve().parent.parent / "jobs"
JOBS_ROOT.mkdir(parents=True, exist_ok=True)


def _new_job_id() -> str:
    """Generate a unique job identifier."""
    return uuid.uuid4().hex[:10]


def start_job(train_config_dict: dict[str, Any]) -> str:
    """Create and launch a new training job.

    The job configuration is persisted to ``jobs/<id>/config.json``, and the
    training worker is launched as a detached subprocess.

    Args:
        train_config_dict: Full training configuration as a dictionary.

    Returns:
        The job ID string.
    """
    job_id = _new_job_id()
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Persist config for the worker to read
    (job_dir / "config.json").write_text(
        json.dumps(train_config_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Initialize status
    (job_dir / "status.json").write_text(
        json.dumps({"phase": "queued", "status": "running"}),
        encoding="utf-8",
    )

    log_path = job_dir / "log.txt"
    cmd = [
        sys.executable,
        "-m",
        "app.train_worker",
        "--job-dir",
        str(job_dir),
    ]
    log_file = open(log_path, "ab", buffering=0)

    popen_kwargs: dict[str, Any] = dict(
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).resolve().parent.parent),
    )

    if platform.system() == "Windows":
        # Fully detach from the current console/session so the worker
        # survives RDP disconnects or terminal closure.
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        # On Linux/macOS, create a new session (similar to nohup) so the
        # worker survives SSH disconnects.
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    (job_dir / "pid.txt").write_text(str(proc.pid), encoding="utf-8")

    return job_id


def _pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID is still running."""
    if psutil is not None:
        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_status(job_id: str) -> dict[str, Any]:
    """Retrieve the current status of a job from its on-disk files.

    Args:
        job_id: The job identifier.

    Returns:
        dict with keys: ``job_id``, ``pid``, ``pid_alive``, ``status``,
        ``phase``, ``step``, ``total_steps``, ``last_loss``,
        ``loss_history``, ``log_tail``, and any fields present in
        the job's ``status.json``.
    """
    job_dir = JOBS_ROOT / job_id
    if not job_dir.exists():
        return {"error": "job_not_found"}

    status_path = job_dir / "status.json"
    status: dict[str, Any] = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.exists()
        else {}
    )

    pid_path = job_dir / "pid.txt"
    pid: Optional[int] = None
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
        except ValueError:
            pid = None

    alive = _pid_alive(pid) if pid is not None else False

    # If the job claims to be running but its PID is gone, mark it crashed
    if status.get("status") == "running" and pid is not None and not alive:
        status["status"] = "crashed_or_stopped"

    # Load recent loss history (last 200 points)
    loss_history: list[dict[str, Any]] = []
    loss_path = job_dir / "loss_history.jsonl"
    if loss_path.exists():
        lines = loss_path.read_text(encoding="utf-8").strip().splitlines()
        loss_history = [json.loads(line) for line in lines[-200:]]

    # Load tail of the training log
    log_tail = ""
    log_path = job_dir / "log.txt"
    if log_path.exists():
        try:
            log_tail = log_path.read_text(
                encoding="utf-8", errors="ignore"
            )[-4000:]
        except Exception:
            log_tail = ""

    status["job_id"] = job_id
    status["pid"] = pid
    status["pid_alive"] = alive
    status["loss_history"] = loss_history
    status["log_tail"] = log_tail

    return status


def stop_job(job_id: str) -> dict[str, Any]:
    """Terminate a running job's subprocess and mark it as stopped.

    Args:
        job_id: The job identifier.

    Returns:
        dict with ``"ok": True`` on success, or ``"error"`` on failure.
    """
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

    # Update status.json to reflect the stopped state
    status_path = job_dir / "status.json"
    data = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.exists()
        else {}
    )
    data.update({"phase": "stopped", "status": "stopped"})
    status_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {"ok": True}


def resume_job(job_id: str) -> str:
    """Resume a stopped job from its last checkpoint.

    The original job directory is left intact; a new job is created with
    ``resume_from_checkpoint`` pointing to the latest checkpoint found.

    Args:
        job_id: The identifier of the job to resume.

    Returns:
        The new job ID.
    """
    old_job_dir = JOBS_ROOT / job_id
    cfg = json.loads((old_job_dir / "config.json").read_text(encoding="utf-8"))

    # Find the latest checkpoint
    ckpt_root = old_job_dir / "checkpoints"
    last_ckpt: Optional[str] = None
    if ckpt_root.exists():
        checkpoints = sorted(
            ckpt_root.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if checkpoints:
            last_ckpt = str(checkpoints[-1])

    cfg["resume_from_checkpoint"] = last_ckpt
    return start_job(cfg)


def list_jobs() -> list[dict[str, Any]]:
    """Return status summaries for all jobs, newest first."""
    jobs: list[dict[str, Any]] = []
    for d in sorted(
        JOBS_ROOT.iterdir(),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        if d.is_dir() and (d / "config.json").exists():
            jobs.append(get_status(d.name))
    return jobs
