# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — Initial Release

### Added
- Hardware auto-detection (CPU, RAM, GPU/VRAM) with profile recommendation
- Curated catalog of Small Language Models (Qwen, Gemma, Phi, TinyLlama, Llama)
- Multi-format dataset upload & conversion (Alpaca, ShareGPT, Dataset Generator)
- LoRA/QLoRA fine-tuning engine (transformers + peft + trl)
- Detached subprocess training with disk-based progress persistence
- Live training monitoring with real-time loss charts
- Checkpoint save/resume per epoch
- LoRA adapter merge with base model
- In-app model serving for quick evaluation
- GGUF export instructions
- Dark/light theme toggle
- Persian (RTL) web-based UI in a native desktop window (pywebview)
- Windows one-click launcher (`run_windows.bat`)
- Linux server launcher (`run_server.sh`) with systemd service template
- MIT License
