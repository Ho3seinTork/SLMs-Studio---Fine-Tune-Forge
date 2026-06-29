# Contributing to SLM Forge

Thank you for your interest in contributing to SLM Forge!

## Development Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/slm-forge/slm-forge.git
   cd slm-forge
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

3. **Install development dependencies**:

   ```bash
   pip install -e ".[dev]"
   ```

   For ML-related development (requires PyTorch installed first):

   ```bash
   pip install -e ".[all]"
   ```

4. **Run tests**:

   ```bash
   pytest
   ```

   With coverage:

   ```bash
   pytest --cov=app --cov-report=html
   ```

## Code Style

SLM Forge uses [Ruff](https://github.com/astral-sh/ruff) for linting and
formatting. Configuration is in `pyproject.toml`.

```bash
# Lint
ruff check app/ tests/

# Format
ruff format app/ tests/
```

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Add tests for any new functionality.
3. Ensure all tests pass (`pytest`).
4. Run the linter (`ruff check .`).
5. Submit a pull request with a clear description.

### Commit Messages

Use conventional commit format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code restructuring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

## Testing Guidelines

- Tests live in the `tests/` directory.
- Use `pytest` fixtures for shared setup.
- Unit tests should not require GPU or internet access.
- ML-dependent tests are skipped when `transformers` is not installed.
- Use `conftest.py` for shared test fixtures.

## Project Architecture

```
app/
├── main.py            Entry point (desktop window launcher)
├── api.py             FastAPI REST routes
├── hardware.py        System detection (no ML deps)
├── models_catalog.py  Static model catalog (no ML deps)
├── dataset_converter.py  Dataset format conversion (no ML deps)
├── trainer.py         LoRA/QLoRA training engine (ML deps)
├── train_worker.py    Subprocess entry point (ML deps)
├── job_manager.py     Process management (minimal deps)
├── exporter.py        Model merging (ML deps)
└── server_inference.py  In-process inference (ML deps)
```

Core modules (`hardware`, `models_catalog`, `dataset_converter`) are designed
to work without ML libraries installed. ML-dependent modules (`trainer`,
`exporter`, `server_inference`) use lazy imports so the application can start
and show the UI even before ML dependencies are installed.

## Questions?

Open an issue on GitHub or start a discussion.
