"""Tests for the job_manager module.

These tests verify job creation, status retrieval, and lifecycle operations
without requiring actual ML training (no GPU/transformers needed).
"""

import json
from pathlib import Path

import pytest

from app import job_manager


def test_jobs_root_exists():
    """The jobs root directory should exist after the module is imported."""
    assert job_manager.JOBS_ROOT.exists()
    assert job_manager.JOBS_ROOT.is_dir()


def test_new_job_id_is_unique():
    """Each generated job ID should be unique."""
    ids = {job_manager._new_job_id() for _ in range(20)}
    assert len(ids) == 20


def test_new_job_id_is_string():
    """Job IDs should be hex strings."""
    jid = job_manager._new_job_id()
    assert isinstance(jid, str)
    assert len(jid) == 10
    assert all(c in "0123456789abcdef" for c in jid)


def test_start_job_creates_directory_and_files():
    """start_job() should create the job directory, config.json, status.json."""
    cfg = {
        "base_model": "test/model",
        "train_file": "/tmp/test.jsonl",
        "eval_file": None,
        "output_dir": "/tmp",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "num_train_epochs": 1.0,
        "per_device_batch_size": 1,
        "grad_accum_steps": 4,
        "max_seq_len": 512,
        "use_4bit": False,
        "precision": "fp32",
        "gradient_checkpointing": False,
        "seed": 42,
        "resume_from_checkpoint": None,
        "hf_token": None,
    }

    # start_job launches a subprocess; it will fail because there's no real
    # training data, but the directory and config files should be created.
    job_id = job_manager.start_job(cfg)

    job_dir = job_manager.JOBS_ROOT / job_id
    assert job_dir.exists()
    assert (job_dir / "config.json").exists()
    assert (job_dir / "status.json").exists()

    # Verify config was persisted correctly
    saved_cfg = json.loads(
        (job_dir / "config.json").read_text(encoding="utf-8")
    )
    assert saved_cfg["base_model"] == "test/model"
    assert saved_cfg["lora_r"] == 8

    # Clean up — the worker likely failed quickly since there's no data
    # Mark the job as stopped so the directory is clean
    status_path = job_dir / "status.json"
    status_path.write_text(
        json.dumps({"phase": "stopped", "status": "stopped"}),
        encoding="utf-8",
    )


def test_get_status_nonexistent_job():
    """get_status() for a nonexistent job should return an error."""
    result = job_manager.get_status("nonexistent_job_id_12345")
    assert "error" in result
    assert result["error"] == "job_not_found"


def test_list_jobs_returns_list():
    """list_jobs() should return a list."""
    jobs = job_manager.list_jobs()
    assert isinstance(jobs, list)


def test_stop_job_nonexistent():
    """stop_job() for a nonexistent PID should return an error."""
    result = job_manager.stop_job("nonexistent_job_id_12345")
    assert "error" in result


def test_resume_job_nonexistent():
    """resume_job() for a nonexistent job should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        job_manager.resume_job("nonexistent_job_id_12345")


def test_pid_alive_for_nonexistent():
    """_pid_alive() should return False for a nonexistent PID."""
    assert job_manager._pid_alive(99999999) is False
