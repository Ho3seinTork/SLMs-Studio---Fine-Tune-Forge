# -*- coding: utf-8 -*-
"""
ورکر مجزای آموزش | Standalone training worker
---------------------------------------------------------------------------
این فایل با دستور:  python -m app.train_worker --job-dir <path>
به عنوان یک پروسه کاملاً مجزا (detached) اجرا می‌شود تا چرخه حیات آموزش به
هیچ پنجره GUI، نشست SSH یا Remote Desktop وابسته نباشد — یعنی اگر ارتباط
قطع شود، این پروسه (که از parent جدا شده) همچنان روی دیسک پیشرفت می‌نویسد.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.trainer import TrainConfig, run_training  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    cfg_path = job_dir / "config.json"
    cfg = TrainConfig.from_json(cfg_path.read_text(encoding="utf-8"))

    pid_file = job_dir / "pid.txt"
    pid_file.write_text(str(__import__("os").getpid()), encoding="utf-8")

    try:
        run_training(cfg, str(job_dir))
    except Exception:
        err_text = traceback.format_exc()
        (job_dir / "error.txt").write_text(err_text, encoding="utf-8")
        status_path = job_dir / "status.json"
        try:
            data = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        except Exception:
            data = {}
        data.update({"phase": "failed", "status": "failed", "error": err_text[-2000:]})
        status_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        sys.exit(1)


if __name__ == "__main__":
    main()
