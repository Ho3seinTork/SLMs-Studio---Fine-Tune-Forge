"""Standalone training worker — entry point for detached subprocesses.

This module is invoked by ``job_manager.py`` as a separate OS process::

    python -m app.train_worker --job-dir <path>

Running training in a fully detached subprocess ensures that the training
lifecycle is independent of any GUI window, SSH session, or Remote Desktop
connection. If the parent process or remote session is terminated, the
training worker continues writing progress to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.trainer import TrainConfig, run_training  # noqa: E402


def main() -> None:
    """Parse arguments, load config, and execute the training run."""
    parser = argparse.ArgumentParser(
        description="SLM Forge standalone training worker"
    )
    parser.add_argument("--job-dir", required=True, help="Job output directory")
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    cfg_path = job_dir / "config.json"

    if not cfg_path.exists():
        print(f"FATAL: config.json not found in {job_dir}", file=sys.stderr)
        sys.exit(1)

    cfg = TrainConfig.from_json(cfg_path.read_text(encoding="utf-8"))

    # Record PID for the job manager to monitor
    pid_file = job_dir / "pid.txt"
    pid_file.write_text(str(os.getpid()), encoding="utf-8")

    try:
        run_training(cfg, str(job_dir))
    except Exception:
        err_text = traceback.format_exc()

        # Write full traceback to disk for post-mortem debugging
        (job_dir / "error.txt").write_text(err_text, encoding="utf-8")

        # Update status.json with failure information
        status_path = job_dir / "status.json"
        try:
            data = (
                json.loads(status_path.read_text(encoding="utf-8"))
                if status_path.exists()
                else {}
            )
        except Exception:
            data = {}

        data.update({
            "phase": "failed",
            "status": "failed",
            "error": err_text[-2000:],
        })
        status_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
