#!/usr/bin/env bash
# اجرای SLM Forge روی سرور (لینوکس) با GPU قدرتمند.
# نکته حیاتی برای جلوگیری از قطع‌شدن با بسته‌شدن نشست SSH/Remote:
#   این اسکریپت را داخل tmux یا screen اجرا کنید، مثلاً:
#     tmux new -s slmforge
#     bash run_server.sh
#   حتی اگر این کار را هم نکنید، خودِ job_manager.py هر job آموزش را به‌صورت
#   یک پروسه کاملاً مجزا (با start_new_session) اجرا می‌کند، یعنی آموزش با
#   قطع همین اسکریپت/SSH هم متوقف نمی‌شود — اما بهتر است خودِ سرور API هم
#   داخل tmux/screen یا یک systemd service بماند تا بتوانید به UI دسترسی داشته باشید.
set -e
cd "$(dirname "$0")"

if [ ! -d venv ]; then
  echo "[1/3] ساخت virtualenv..."
  python3 -m venv venv
fi
source venv/bin/activate

echo "[2/3] نصب وابستگی‌ها (torch با CUDA را طبق README جدا نصب کنید)..."
pip install -r requirements.txt

echo "[3/3] اجرای سرور API روی پورت 8765 ..."
echo "آدرس دسترسی: http://<IP-سرور>:8765  (یا با SSH tunnel: ssh -L 8765:localhost:8765 user@server)"
python3 -m uvicorn app.api:app --host 0.0.0.0 --port 8765
