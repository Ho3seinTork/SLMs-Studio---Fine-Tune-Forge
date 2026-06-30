# SLM Forge — Small Language Model Fine-tuning Studio

<img width="1918" height="1078" alt="image" src="https://github.com/user-attachments/assets/37f87614-7c18-494a-9486-962d10bdfa51" />

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A desktop application for fine-tuning **Small Language Models** (SLMs) — models
like Qwen, Gemma, Phi, TinyLlama, and Llama in the <10B parameter range — using
**LoRA** and **QLoRA** with a graphical interface.

SLM Forge runs on both low-resource laptops (CPU-only) and powerful GPU servers,
with training executed as detached OS processes that survive SSH/RDP disconnects.

<p align="center">
  <em>UI shown in dark theme — automatically adapts to light/dark mode.</em>
</p>

---

## Features

- **Hardware auto-detection** — Detects RAM, CPU cores, GPU/VRAM and recommends
  optimal hyperparameters for your machine.
- **Curated model catalog** — Pre-configured list of popular SLMs with hardware
  compatibility indicators.
- **Multi-format dataset support** — Accepts Alpaca, ShareGPT, and custom
  Dataset Generator JSON formats with automatic format detection.
- **LoRA/QLoRA fine-tuning** — Full support for LoRA (fp32/fp16/bf16) and 4-bit
  QLoRA quantization via `bitsandbytes`.
- **Detached training** — Each training job runs as a fully independent OS
  process. Close the GUI, disconnect SSH/RDP — training continues and writes
  progress to disk.
- **Live monitoring** — Real-time loss charts, step progress, and log tails
  in the UI (updates every 4 seconds).
- **Checkpoint & resume** — Training checkpoints are saved at each epoch.
  Stopped jobs can be resumed from the last checkpoint.
- **LoRA merge** — Merge trained adapters with the base model to produce a
  standalone model directory.
- **In-app testing** — Load fine-tuned models directly in the UI for quick
  evaluation prompts.
- **GGUF export guide** — Built-in instructions for converting to GGUF for
  llama.cpp / Ollama deployment.

---

## Quick Start

### Windows (double-click)

1. Install Python 3.10+ from [python.org](https://www.python.org/downloads/)
   (check "Add python.exe to PATH" during installation).
2. Double-click `run_windows.bat`.
3. The script creates a virtual environment, installs dependencies, and
   launches the application.

### Linux/macOS Server

```bash
git clone https://github.com/slm-forge/slm-forge.git
cd slm-forge

# Install PyTorch for your hardware first:
# CPU:  pip install torch --index-url https://download.pytorch.org/whl/cpu
# CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
python -m app.main
```

For headless servers, run just the API server and access the UI from your
browser via SSH tunnel:

```bash
python -m uvicorn app.api:app --host 0.0.0.0 --port 8765
# From your local machine:
ssh -L 8765:localhost:8765 user@server-ip
# Then open http://127.0.0.1:8765 in your browser
```

---

## Usage Guide

### 1. Hardware Detection
The top card displays your system specs (OS, CPU, RAM, GPU) and recommends
a training profile. Hyperparameters are pre-filled based on this profile.

### 2. Dataset Upload
Upload a JSON dataset file. Supported formats:
- **Dataset Generator** output (JSON array with `status`, `conversation` fields)
- **Alpaca** (`instruction`, `input`, `output`)
- **ShareGPT** (`conversations` with `from`/`value`)

Only samples with `status: "success"` (Dataset Generator) or valid
conversation pairs are used.

### 3. Model Selection
Choose a base model from the catalog. Models marked with a "Gated" badge
require accepting the license on Hugging Face and providing an access token.

### 4. Hyperparameter Configuration
LoRA parameters (rank, alpha, dropout, learning rate, batch size, epochs,
sequence length) are pre-filled from the hardware profile but can be
customized.

### 5. Start Training
Click "Start Fine-tuning". A new Job appears in the main panel with:
- Live progress bar
- Real-time loss chart
- Current phase and step counter
- Stop / Resume buttons

### 6. Export & Merge
After training completes, use the Export section to:
- **Merge** the LoRA adapter with the base model → standalone model directory
- **Convert to GGUF** — follow the built-in instructions for llama.cpp/Ollama

### 7. In-App Testing
Load the merged model and run test prompts directly in the UI.

---

## Hardware Requirements

| Hardware Tier | Profile | Recommended Models | Expected Speed |
|---|---|---|---|
| CPU only, 8 GB RAM | `laptop_cpu` | Qwen2.5-0.5B, TinyLlama-1.1B | Very slow (hours per epoch) |
| Laptop GPU, 4–6 GB VRAM | `laptop_gpu_small` | Up to ~4B params with QLoRA | Moderate |
| Server GPU, ≥16 GB VRAM | `server_gpu` | Up to ~14B params | Fast |

> **Note**: Fine-tuning language models on CPU-only systems is inherently slow.
> SLM Forge does not work miracles — it optimizes what your hardware can do,
> but a GPU is strongly recommended for productive use.

---

## Project Structure

```
slm_forge/
├── app/                          # Python package
│   ├── __init__.py               # Version info
│   ├── main.py                   # Entry point (desktop window)
│   ├── api.py                    # FastAPI REST backend
│   ├── hardware.py               # System detection & profile advisor
│   ├── models_catalog.py         # Curated SLM catalog
│   ├── dataset_converter.py      # Multi-format dataset converter
│   ├── trainer.py                # LoRA/QLoRA training engine
│   ├── train_worker.py           # Standalone subprocess worker
│   ├── job_manager.py            # Job lifecycle (start/stop/resume/monitor)
│   ├── exporter.py               # LoRA merge & GGUF export guide
│   └── server_inference.py       # Lightweight model serving for testing
├── ui/                           # Web frontend (Persian RTL)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/                        # Pytest test suite
├── jobs/                         # Runtime job data (gitignored)
├── .github/workflows/ci.yml      # CI pipeline
├── pyproject.toml                # Project metadata & tool config
├── requirements.txt              # Python dependencies
├── run_windows.bat               # Windows one-click launcher
├── run_server.sh                 # Linux server launcher
├── slmforge.service.example      # systemd service template
├── LICENSE
├── README.md
└── CONTRIBUTING.md
```

---

## How Detached Training Works

SLM Forge uses a **two-layer resilience strategy** to ensure training survives
disconnections:

1. **Primary layer**: Each training job is launched via `train_worker.py` as a
   detached OS subprocess:
   - **Linux/macOS**: `start_new_session=True` (equivalent to `nohup`)
   - **Windows**: `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` flags
   
   The worker writes progress to disk files (`status.json`,
   `loss_history.jsonl`, `log.txt`). Even if you close the GUI, browser,
   or SSH session, the worker continues.

2. **Recommended additional layer**: Run the API server itself inside
   `tmux`/`screen` (Linux) or as a systemd service so the UI remains
   accessible across sessions. A sample systemd unit is provided in
   `slmforge.service.example`.

---

## API Reference

The FastAPI backend exposes the following endpoints (all under `/api/`):

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/hardware` | Hardware specs & recommended profile |
| GET | `/api/models` | Model catalog filtered by hardware |
| POST | `/api/dataset/upload` | Upload & preview a dataset |
| POST | `/api/train/start` | Start a fine-tuning job |
| POST | `/api/train/{id}/stop` | Stop a running job |
| POST | `/api/train/{id}/resume` | Resume from checkpoint |
| GET | `/api/jobs` | List all jobs |
| GET | `/api/jobs/{id}/status` | Get job status & loss history |
| POST | `/api/export/merge` | Merge LoRA adapter with base model |
| GET | `/api/export/gguf-instructions` | GGUF conversion guide |
| POST | `/api/serve/start` | Load model for inference |
| POST | `/api/serve/stop` | Unload model |
| GET | `/api/serve/health` | Check if model is loaded |
| POST | `/api/serve/generate` | Generate text from loaded model |

---

## Known Limitations

- **`bitsandbytes` on Windows**: The package (required for QLoRA 4-bit
  quantization) may require Visual C++ Redistributable. If it fails to load,
  disable the QLoRA checkbox — standard LoRA works without it but uses more
  memory.
- **Gated models**: Gemma, Llama, and other gated models require accepting the
  license on [huggingface.co](https://huggingface.co) and creating a read-only
  access token.
- **No built-in GGUF export**: Due to the complexity of C++ compilation
  (llama.cpp), SLM Forge provides step-by-step instructions instead of
  automatic GGUF conversion.

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for
guidelines on setting up a development environment, running tests, and
submitting pull requests.

---

## License

MIT — see [LICENSE](LICENSE) for full text.

---

## Acknowledgments

Built on the Hugging Face ecosystem:
[transformers](https://github.com/huggingface/transformers),
[peft](https://github.com/huggingface/peft),
[trl](https://github.com/huggingface/trl),
[datasets](https://github.com/huggingface/datasets).

Desktop window provided by [pywebview](https://github.com/r0x0r/pywebview).

# SLM Forge — استودیوی فاین‌تیونینگ مدل‌های زبانی کوچک

<img width="1918" height="1078" alt="2" src="https://github.com/user-attachments/assets/362cc9f9-b35e-40e4-8578-63427301127a" />


اپلیکیشنی با رابط گرافیکی زیبا (FastAPI + UI وب محلی، باز شده در یک پنجره دسکتاپ با pywebview) برای
فاین‌تیون‌کردن مدل‌های زبانی کوچک (SLM مثل Gemma، Qwen، Phi، TinyLlama، Llama) با تکنیک
LoRA/QLoRA، روی دیتاستی که اپ «Dataset Generator» قبلی ساخته است. هم روی لپ‌تاپ ضعیف
و هم روی سرور قدرتمند (با GPU) کار می‌کند، و آموزش به‌صورت یک پروسه کاملاً مجزا اجرا می‌شود
تا با قطع Remote Desktop/SSH متوقف نشود.

## ⚠️ واقعیتِ سخت‌افزاری — قبل از هر چیز بخوانید

این برنامه «معجزه» نمی‌کند: فاین‌تیون‌کردن مدل زبانی روی لپ‌تاپ با ۸ گیگ رم و **بدون GPU**
کار می‌کند ولی **بسیار کند** است (می‌تواند چند ساعت تا یک شب طول بکشد، حتی برای مدل‌های کوچک
و دیتاست کوچک). برنامه به‌صورت خودکار سخت‌افزار شما را تشخیص می‌دهد و:

| سخت‌افزار شما | پروفایل | مدل پیشنهادی |
|---|---|---|
| فقط CPU، ۸ گیگ رم | `laptop_cpu` | Qwen2.5-0.5B یا TinyLlama-1.1B |
| لپ‌تاپ با GPU ۴-۶ گیگ VRAM | `laptop_gpu_small` | تا حدود Gemma-2-2B با QLoRA |
| سرور با GPU ≥۱۶ گیگ VRAM | `server_gpu` | تا حدود ۹-۱۴ میلیارد پارامتر |

اگر فقط می‌خواهید سریع تست کنید که همه‌چیز کار می‌کند، **Qwen2.5-0.5B-Instruct** را با دیتاست کوچک
(۱۰-۲۰ نمونه) انتخاب کنید.

## نصب روی ویندوز (لپ‌تاپ)

1. پایتون ۳.10 یا بالاتر را از python.org نصب کنید (هنگام نصب تیک «Add to PATH» را بزنید).
2. روی `run_windows.bat` دوبار کلیک کنید. این فایل به‌صورت خودکار:
   - یک محیط مجازی (`venv`) می‌سازد،
   - نسخه CPU از torch را نصب می‌کند (اگر کارت گرافیک NVIDIA و درایور CUDA دارید، می‌توانید بعداً
     طبق دستور رسمی pytorch.org نسخه CUDA را جای‌گزین کنید تا سریع‌تر شود)،
   - بقیه پکیج‌ها را از `requirements.txt` نصب می‌کند،
   - برنامه را اجرا می‌کند و یک پنجره دسکتاپ باز می‌شود.

اگر پنجره دسکتاپ باز نشد (مثلاً pywebview درست نصب نشد)، برنامه به‌جای آن آدرس
`http://127.0.0.1:8765` را در ترمینال چاپ می‌کند — همان آدرس را در مرورگر باز کنید.

## نصب و اجرا روی سرور (لینوکس با GPU قدرتمند)

```bash
git clone/کپی این پوشه روی سرور
cd slm_forge
bash run_server.sh
```

این اسکریپت فقط بک‌اند API را (بدون پنجره گرافیکی) روی پورت `8765` بالا می‌آورد. برای دیدن
رابط کاربری از مرورگر خودتان، یک SSH tunnel بزنید:

```bash
ssh -L 8765:localhost:8765 user@server-ip
```

و بعد در مرورگر خودتان `http://127.0.0.1:8765` را باز کنید.

### چطور با قطع Remote Desktop / SSH متوقف نمی‌شود؟

دو لایه محافظت وجود دارد:

1. **مهم‌ترین لایه:** هر «Job» آموزش، توسط `job_manager.py` به‌صورت یک **پروسه سیستمی کاملاً مجزا**
   اجرا می‌شود (روی لینوکس با `start_new_session=True` شبیه nohup، روی ویندوز با
   `DETACHED_PROCESS`). یعنی حتی اگر کل برنامه/مرورگر/نشست SSH را ببندید، خود فرآیند آموزش
   روی دیسک (`status.json`, `loss_history.jsonl`, `log.txt`) پیشرفت می‌نویسد و ادامه می‌دهد.
   با باز کردن دوباره برنامه، از همان `jobs/` خوانده می‌شود.
2. **لایه توصیه‌شده اضافه:** خودِ سرور API را هم بهتر است داخل `tmux`/`screen` یا به‌صورت
   سرویس systemd اجرا کنید (نمونه فایل در `slmforge.service.example` آمده) تا رابط کاربری هم
   همیشه در دسترس بماند.

نکته درباره **Remote Desktop ویندوز**: اگر فقط نشست را **Disconnect** کنید (نه Log off)،
معمولاً پروسه‌ها روی سرور ادامه پیدا می‌کنند. اگر Log off کنید، ویندوز پروسه‌های آن کاربر را
می‌بندد — برای اطمینان کامل، `python -m app.main` را از طریق Task Scheduler با گزینه
«Run whether user is logged on or not» اجرا کنید، یا با ابزاری مثل NSSM آن را به یک
Windows Service تبدیل کنید.

## نحوه کار با برنامه

1. **سخت‌افزار**: کارت بالای صفحه به‌صورت خودکار RAM/CPU/GPU را نشان می‌دهد و هایپرپارامترها
   را بر همان اساس از قبل پر می‌کند (قابل ویرایش).
2. **دیتاست**: فایل `dataset.json` که اپ Dataset Generator ساخته را آپلود کنید (فرمت‌های
   Alpaca و ShareGPT هم پشتیبانی می‌شوند). فقط نمونه‌های با `status: "success"` استفاده می‌شوند.
3. **مدل پایه**: یکی از مدل‌های کاتالوگ را انتخاب کنید (مدل‌های Gated مثل Gemma/Llama نیاز به
   پذیرفتن لایسنس در صفحه مدل در Hugging Face و وارد کردن یک «Access Token» دارند).
4. **هایپرپارامتر LoRA**: مقادیر پیشنهادی از قبل پر شده؛ در صورت نیاز تغییر دهید.
5. روی **«شروع فاین‌تیونینگ»** بزنید. یک Job جدید ساخته می‌شود و در پایین صفحه با نمودار loss
   زنده و درصد پیشرفت نمایش داده می‌شود (هر ۴ ثانیه به‌روزرسانی می‌شود).
6. می‌توانید هر Job را **متوقف** کنید یا بعداً از آخرین چک‌پوینت **ادامه** دهید.
7. پس از پایان، در بخش **خروجی‌گیری**، آداپتور LoRA را با مدل پایه **ادغام** کنید تا یک مدل
   کامل و مستقل بسازید.
8. همان مدل ادغام‌شده را با دکمه **«سوار کردن مدل»** بارگذاری کرده و مستقیماً در همین صفحه تست کنید
   («مدل روی سرور سوار می‌شود»).

## خروجی GGUF برای llama.cpp / Ollama (اختیاری)

این برنامه به‌صورت خودکار GGUF نمی‌سازد (نیاز به build کردن llama.cpp دارد که خارج از scope
یک اپ پایتونی است)، اما دکمه «راهنمای خروجی GGUF» در UI، دستورات دقیق گام‌به‌گام را نشان می‌دهد.

## ساختار پروژه

```
slm_forge/
├── app/
│   ├── hardware.py          تشخیص RAM/CPU/GPU و پیشنهاد پروفایل
│   ├── models_catalog.py    فهرست مدل‌های SLM پشتیبانی‌شده
│   ├── dataset_converter.py تبدیل dataset.json به فرمت آموزش (JSONL)
│   ├── trainer.py           هسته آموزش LoRA/QLoRA (transformers+peft+trl)
│   ├── train_worker.py      ورکر مجزا که هر Job در آن واقعاً اجرا می‌شود
│   ├── job_manager.py       استارت/توقف/ادامه/پایش Jobها به‌صورت پروسه مجزا
│   ├── exporter.py          ادغام LoRA با مدل پایه
│   ├── server_inference.py  سرو مدل برای تست سریع
│   ├── api.py                FastAPI: تمام endpoint ها + سرو فایل‌های UI
│   └── main.py                نقطه ورود: سرور + پنجره دسکتاپ (pywebview)
├── ui/                       index.html / style.css / app.js
├── jobs/                     (در زمان اجرا ساخته می‌شود) داده/چک‌پوینت/لاگ هر Job
├── requirements.txt
├── run_windows.bat
├── run_server.sh
└── slmforge.service.example
```

## نکات امنیتی

- توکن Hugging Face فقط در حافظه مرورگر شما (در همین نشست) نگه داشته می‌شود و مستقیماً به
  Hugging Face ارسال می‌شود؛ روی هیچ سروری ذخیره نمی‌گردد.
- این برنامه فقط روی `127.0.0.1` (یا با `--host 0.0.0.0` روی شبکه داخلی سرور) اجرا می‌شود.
  اگر روی سروری با IP عمومی اجرا می‌کنید، حتماً پشت SSH tunnel یا VPN/فایروال بمانید — این UI
  هیچ احراز هویتی ندارد.

## محدودیت‌های شناخته‌شده / صادقانه

- کد این پروژه با دقت و بر اساس API های استاندارد `transformers`/`peft`/`trl` نوشته شده، اما در
  محیط توسعه من (Claude) امکان دانلود واقعی مدل از Hugging Face یا اجرای یک training step واقعی
  روی GPU وجود نداشت؛ پس فقط بخش‌های بدون نیاز به اینترنت/GPU (تشخیص سخت‌افزار، تبدیل دیتاست)
  واقعاً اجرا و تست شده‌اند. پیشنهاد می‌کنم اولین اجرا را با کوچک‌ترین مدل (Qwen2.5-0.5B) و یک
  دیتاست کوچک تست کنید تا از صحت کامل مسیر روی سیستم خودتان مطمئن شوید.
- `bitsandbytes` (لازم برای QLoRA ۴-بیتی) روی ویندوز گاهی به Visual C++ Redistributable نیاز دارد؛
  اگر بارگذاری نشد، گزینه «QLoRA (4bit)» را خاموش بگذارید (آموزش معمولی LoRA بدون آن هم کار می‌کند،
  فقط حافظه بیشتری می‌خواهد).
- مدل‌های Gated (Gemma، Llama) نیاز دارند ابتدا در صفحه مدل روی huggingface.co لایسنس را بپذیرید
  و یک Access Token با دسترسی Read بسازید.
