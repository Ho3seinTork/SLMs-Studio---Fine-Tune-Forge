"""Application entry point for SLM Forge.

Starts a local FastAPI server in a background thread and opens the desktop
UI, preferring a native pywebview window on Windows/macOS/Linux with a
browser fallback.
"""

from __future__ import annotations

import sys
import threading
import time
import webbrowser
from pathlib import Path

# Ensure the project root is on sys.path so ``app.*`` imports work
# regardless of how the script was invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402
from app.api import app  # noqa: E402

HOST = "127.0.0.1"
PORT = 8765


def _run_server() -> None:
    """Start the FastAPI server (blocking, runs in a daemon thread)."""
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


def main() -> None:
    """Launch the server and open the desktop window."""
    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()
    time.sleep(1.5)  # Allow the server to bind before opening the window

    url = f"http://{HOST}:{PORT}"

    try:
        import webview  # noqa: WPS433

        webview.create_window(
            "SLM Forge — Small Language Model Fine-tuning Studio",
            url,
            width=1400,
            height=880,
            min_size=(1080, 680),
        )
        webview.start()
    except ImportError:
        print(
            f"pywebview is not installed. "
            f"Opening default browser at {url}"
        )
        webbrowser.open(url)
        print("Close this terminal window to stop the server (Ctrl+C).")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
