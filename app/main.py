# -*- coding: utf-8 -*-
"""
نقطه ورود برنامه | Application entrypoint
---------------------------------------------------------------------------
یک سرور FastAPI محلی را در یک ترد جانبی بالا می‌آورد و سپس یا یک پنجره
دسکتاپ بومی (با pywebview) باز می‌کند یا (اگر pywebview نصب نباشد) مرورگر
پیش‌فرض سیستم را باز می‌کند. این کار باعث می‌شود برنامه روی ویندوز هم با
دابل‌کلیک شبیه یک اپلیکیشن واقعی دسکتاپ اجرا شود.
"""
import sys
import time
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402
from app.api import app  # noqa: E402

HOST = "127.0.0.1"
PORT = 8765


def _run_server():
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


def main():
    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()
    time.sleep(1.5)
    url = f"http://{HOST}:{PORT}"

    try:
        import webview
        webview.create_window(
            "SLM Forge — استودیوی فاین‌تیونینگ مدل‌های زبانی کوچک",
            url, width=1400, height=880, min_size=(1080, 680),
        )
        webview.start()
    except ImportError:
        print(f"pywebview نصب نیست — مرورگر پیش‌فرض روی {url} باز می‌شود.")
        webbrowser.open(url)
        print("برای خروج از برنامه، این پنجره ترمینال را ببندید (Ctrl+C).")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
