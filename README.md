# SLM Forge — Small Language Model Fine-tuning Studio

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

---

<div align="center">

# 🔥 SLM Forge — کارگاه تنظیم‌دقیق مدل‌های زبانی کوچک

<br>

![Version](https://img.shields.io/badge/version-0.1.0-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Code Style](https://img.shields.io/badge/Code%20Style-ruff-000000?style=for-the-badge)

<br>

**یک برنامهٔ دسکتاپ برای تنظیم‌دقیق مدل‌های زبانی کوچک (SLM) — مدل‌هایی مثل Qwen، Gemma، Phi، TinyLlama و Llama در بازهٔ کمتر از ۱۰ میلیارد پارامتر — با استفاده از LoRA و QLoRA و یک رابط گرافیکی ساده.**

<br>

✨ **SLM Forge هم روی لپ‌تاپ‌های کم‌منبع (فقط CPU) و هم روی سرورهای قدرتمند GPU اجرا می‌شود** — با قابلیت اجرای آموزش به‌عنوان فرآیندهای جداگانه از سیستم‌عامل که حتی پس از قطع SSH/RDP هم به کار خود ادامه می‌دهند.

<br>

![UI Preview](https://via.placeholder.com/800x400/1e1e2e/ffffff?text=UI+Preview+%F0%9F%8E%AF)

</div>

---

## ✨ ویژگی‌های کلیدی

| 🎯 ویژگی | 📖 توضیح |
|:---|:---|
| **تشخیص خودکار سخت‌افزار** | تشخیص RAM، هسته‌های CPU، GPU/VRAM و پیشنهاد هایپرپارامترهای بهینه برای سیستم شما. |
| **کاتالوگ مدل‌های آماده** | لیست پیکربندی‌شده از مدل‌های محبوب SLM با نشانگرهای سازگاری با سخت‌افزار. |
| **پشتیبانی از فرمت‌های متنوع دیتاست** | پذیرش فرمت‌های Alpaca، ShareGPT و JSON خروجی Dataset Generator با تشخیص خودکار فرمت. |
| **تنظیم‌دقیق LoRA/QLoRA** | پشتیبانی کامل از LoRA (fp32/fp16/bf16) و کوانتیزاسیون ۴ بیتی QLoRA با استفاده از `bitsandbytes`. |
| **آموزش جداشده (Detached)** | هر کار آموزشی به‌عنوان یک فرآیند کاملاً مستقل از سیستم‌عامل اجرا می‌شود. رابط گرافیکی را ببندید، SSH/RDP را قطع کنید — آموزش ادامه می‌یابد و پیشرفت را روی دیسک ذخیره می‌کند. |
| **مانیتورینگ زنده** | نمودار Loss لحظه‌ای، پیشرفت مرحله و لاگ‌های به‌روز در رابط کاربری (هر ۴ ثانیه یک بار به‌روزرسانی). |
| **چک‌پوینت و ادامه** | چک‌پوینت‌های آموزش در هر ایپاک ذخیره می‌شوند. کارهای متوقف‌شده قابل ادامه از آخرین چک‌پوینت هستند. |
| **ادغام LoRA** | ادغام اداپترهای آموزش‌دیده با مدل پایه برای تولید یک دایرکتوری مدل مستقل. |
| **تست درون‌برنامه‌ای** | بارگذاری مدل‌های تنظیم‌دقیق‌شده به‌طور مستقیم در رابط کاربری برای ارزیابی سریع با پرامپت‌های آزمایشی. |
| **راهنمای خروجی GGUF** | دستورالعمل‌های داخلی برای تبدیل به فرمت GGUF جهت استقرار در `llama.cpp` / `Ollama`. |

---

## 🚀 شروع سریع

### 💻 ویندوز (دوبار کلیک)

1. پایتون ۳.۱۰ یا بالاتر را از [python.org](https://www.python.org) نصب کنید. (در حین نصب، گزینهٔ **«Add python.exe to PATH»** را تیک بزنید.)
2. روی فایل `run_windows.bat` دوبار کلیک کنید.
3. اسکریپت به‌طور خودکار یک محیط مجازی ایجاد می‌کند، وابستگی‌ها را نصب کرده و برنامه را اجرا می‌نماید.

### 🐧 لینوکس / macOS (سرور)

```bash
git clone https://github.com/slm-forge/slm-forge.git
cd slm-forge

# ابتدا PyTorch مناسب سخت‌افزار خود را نصب کنید:
# CPU:  pip install torch --index-url https://download.pytorch.org/whl/cpu
# CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
python -m app.main
```

### 🌐 سرور بدون رابط گرافیکی (Headless)

برای سرورهای بدون نمایشگر، فقط سرور API را اجرا کنید و از طریق SSH Tunnel به رابط کاربری در مرورگر دسترسی داشته باشید:

```bash
python -m uvicorn app.api:app --host 0.0.0.0 --port 8765

# از کامپیوتر محلی خود:
ssh -L 8765:localhost:8765 user@server-ip
# سپس در مرورگر باز کنید: http://127.0.0.1:8765
```

---

## 📖 راهنمای استفاده

### ۱. تشخیص سخت‌افزار
کارت بالای صفحه مشخصات سیستم شما (سیستم‌عامل، CPU، RAM، GPU) را نمایش می‌دهد و یک پروفایل آموزشی توصیه می‌کند. هایپرپارامترها بر اساس این پروفایل به‌طور خودکار پر می‌شوند.

### ۲. آپلود دیتاست
یک فایل JSON دیتاست آپلود کنید. فرمت‌های پشتیبانی‌شده:
- **خروجی Dataset Generator** (آرایه JSON با فیلدهای `status` و `conversation`)
- **Alpaca** (`instruction`, `input`, `output`)
- **ShareGPT** (`conversations` با `from`/`value`)
> فقط نمونه‌های با `status: "success"` (در Dataset Generator) یا جفت‌های گفتگوی معتبر استفاده می‌شوند.

### ۳. انتخاب مدل
یک مدل پایه از کاتالوگ انتخاب کنید. مدل‌های دارای نشان **"Gated"** نیاز به پذیرش مجوز در Hugging Face و ارائه توکن دسترسی دارند.

### ۴. پیکربندی هایپرپارامترها
پارامترهای LoRA (رتبه، آلفا، نرخ حذف، نرخ یادگیری، اندازه بسته، تعداد ایپاک، طول دنباله) بر اساس پروفایل سخت‌افزاری پر می‌شوند اما قابل شخصی‌سازی هستند.

### ۵. شروع آموزش
روی دکمهٔ **«شروع تنظیم‌دقیق»** کلیک کنید. یک کار جدید در پنل اصلی ظاهر می‌شود با:
- نوار پیشرفت زنده
- نمودار Loss لحظه‌ای
- مرحله فعلی و شمارنده گام
- دکمه‌های توقف / ادامه

### ۶. خروجی و ادغام
پس از اتمام آموزش، از بخش **خروجی** استفاده کنید تا:
- اداپتر LoRA را با مدل پایه ادغام کنید ← دایرکتوری مدل مستقل
- به GGUF تبدیل کنید — از دستورالعمل‌های داخلی برای `llama.cpp`/`Ollama` پیروی کنید.

### ۷. تست درون‌برنامه‌ای
مدل ادغام‌شده را بارگذاری کنید و پرامپت‌های آزمایشی را مستقیماً در رابط کاربری اجرا کنید.

---

## ⚙️ نیازمندی‌های سخت‌افزاری

| سطح سخت‌افزار | پروفایل | مدل‌های توصیه‌شده | سرعت تخمینی |
|:---|:---|:---|:---|
| 💻 فقط CPU، ۸ گیگابایت RAM | `laptop_cpu` | Qwen2.5-0.5B، TinyLlama-1.1B | بسیار کند (ساعت‌ها به ازای هر ایپاک) |
| 🎮 لپ‌تاپ GPU، ۴–۶ گیگابایت VRAM | `laptop_gpu_small` | تا حدود ۴ میلیارد پارامتر با QLoRA | متوسط |
| 🖥️ سرور GPU، ≥۱۶ گیگابایت VRAM | `server_gpu` | تا حدود ۱۴ میلیارد پارامتر | سریع |

> ⚠️ توجه: تنظیم‌دقیق مدل‌های زبانی روی سیستم‌های فقط-CPU ذاتاً کند است. SLM Forge معجزه نمی‌کند — کاری که سخت‌افزار شما می‌تواند انجام دهد را بهینه می‌کند. اما برای استفادهٔ مفید، حتماً توصیه می‌شود از GPU استفاده کنید.

---

## 🧱 ساختار پروژه

```
slm_forge/
├── app/                          # بستهٔ پایتون
│   ├── __init__.py               # اطلاعات نسخه
│   ├── main.py                   # نقطهٔ ورود (پنجرهٔ دسکتاپ)
│   ├── api.py                    # باطن REST FastAPI
│   ├── hardware.py               # تشخیص سیستم & مشاور پروفایل
│   ├── models_catalog.py         # کاتالوگ مدل‌های SLM
│   ├── dataset_converter.py      # تبدیل‌کنندهٔ فرمت‌های چندگانه دیتاست
│   ├── trainer.py                # موتور آموزش LoRA/QLoRA
│   ├── train_worker.py           # کارگر پردازش جداگانه
│   ├── job_manager.py            # چرخهٔ حیات کار (شروع/توقف/ادامه/مانیتور)
│   ├── exporter.py               # ادغام LoRA & راهنمای خروجی GGUF
│   └── server_inference.py       # سرویس‌دهی سبک مدل برای تست
├── ui/                           # رابط وب (RTL فارسی)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/                        # مجموعه تست Pytest
├── jobs/                         # داده‌های زمان اجرای کارها (gitignored)
├── .github/workflows/ci.yml      # خط لولهٔ CI
├── pyproject.toml                # متادیتا و تنظیمات ابزار
├── requirements.txt              # وابستگی‌های پایتون
├── run_windows.bat               # راه‌انداز یک‌کلیکی ویندوز
├── run_server.sh                 # راه‌انداز سرور لینوکس
├── slmforge.service.example      # قالب سرویس systemd
├── LICENSE
├── README.md
└── CONTRIBUTING.md
```

---

## 🔧 نحوهٔ عملکرد آموزش جداشده (Detached Training)

SLM Forge از یک استراتژی مقاومت دو-لایه استفاده می‌کند تا اطمینان حاصل کند آموزش پس از قطع اتصال نیز ادامه می‌یابد:

1. **لایهٔ اول**: هر کار آموزشی از طریق `train_worker.py` به‌عنوان یک زیرفرایند جداگانه از سیستم‌عامل راه‌اندازی می‌شود:
   - در لینوکس/macOS: `start_new_session=True` (معادل `nohup`)
   - در ویندوز: پرچم‌های `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`
   
   کارگر پیشرفت را در فایل‌های روی دیسک ذخیره می‌کند (`status.json`، `loss_history.jsonl`، `log.txt`). حتی اگر رابط گرافیکی، مرورگر یا جلسهٔ SSH را ببندید، کارگر به کار خود ادامه می‌دهد.

2. **لایهٔ دوم (توصیه‌شده)**: خود سرور API را داخل `tmux`/`screen` (در لینوکس) یا به‌عنوان یک سرویس systemd اجرا کنید تا رابط کاربری در طول جلسات مختلف در دسترس بماند. یک فایل واحد systemd نمونه در `slmforge.service.example` ارائه شده است.

---

## 📡 مرجع API

باطن FastAPI نقاط پایانی زیر را ارائه می‌دهد (همگی زیر `/api/`):

| متد | نقطه‌پایانی | توضیح |
|:---|:---|:---|
| `GET` | `/api/hardware` | مشخصات سخت‌افزار و پروفایل توصیه‌شده |
| `GET` | `/api/models` | کاتالوگ مدل‌ها بر اساس سخت‌افزار |
| `POST` | `/api/dataset/upload` | آپلود و پیش‌نمایش دیتاست |
| `POST` | `/api/train/start` | شروع یک کار تنظیم‌دقیق |
| `POST` | `/api/train/{id}/stop` | توقف کار در حال اجرا |
| `POST` | `/api/train/{id}/resume` | ادامه از آخرین چک‌پوینت |
| `GET` | `/api/jobs` | لیست همهٔ کارها |
| `GET` | `/api/jobs/{id}/status` | دریافت وضعیت کار و تاریخچه Loss |
| `POST` | `/api/export/merge` | ادغام اداپتر LoRA با مدل پایه |
| `GET` | `/api/export/gguf-instructions` | راهنمای تبدیل GGUF |
| `POST` | `/api/serve/start` | بارگذاری مدل برای استنتاج |
| `POST` | `/api/serve/stop` | تخلیه مدل از حافظه |
| `GET` | `/api/serve/health` | بررسی بارگذاری مدل |
| `POST` | `/api/serve/generate` | تولید متن از مدل بارگذاری‌شده |

---

## ⚠️ محدودیت‌های شناخته‌شده

- **`bitsandbytes` در ویندوز**: این پکیج (برای QLoRA ۴ بیتی) ممکن است به **Visual C++ Redistributable** نیاز داشته باشد. اگر بارگذاری آن با خطا مواجه شد، تیک QLoRA را بردارید — LoRA استاندارد بدون آن کار می‌کند اما حافظهٔ بیشتری مصرف می‌کند.

- **مدل‌های Gated**: مدل‌هایی مانند Gemma، Llama و دیگر مدل‌های دارای محدودیت نیاز به پذیرش مجوز در `huggingface.co` و ساخت یک توکن دسترسی read-only دارند.

- **خروجی GGUF خودکار وجود ندارد**: به دلیل پیچیدگی کامپایل C++ (در `llama.cpp`)، SLM Forge به‌جای تبدیل خودکار GGUF، دستورالعمل‌های گام‌به‌گام ارائه می‌دهد.

---

## 🤝 مشارکت

مشارکت‌ها خوش‌آمد هستند! لطفاً فایل **CONTRIBUTING.md** را برای راهنمایی در مورد راه‌اندازی محیط توسعه، اجرای تست‌ها و ارسال درخواست‌های Pull مطالعه کنید.

---

## 📜 مجوز

این پروژه تحت مجوز **MIT** منتشر شده است — برای جزئیات کامل به فایل `LICENSE` مراجعه کنید.

---

## 🙏 قدردانی

ساخته شده بر روی اکوسیستم Hugging Face:  
🧠 `transformers` • 🔧 `peft` • 🎯 `trl` • 📦 `datasets`

پنجرهٔ دسکتاپ توسط `pywebview` فراهم شده است.

---

<div align="center">

**🌟 اگر از SLM Forge استفاده می‌کنید، به آن ستاره ⭐ بدهید!**

</div>