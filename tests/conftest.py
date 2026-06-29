"""Test suite configuration and shared fixtures."""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def sample_alpaca_data() -> list[dict[str, Any]]:
    """Return a minimal Alpaca-format dataset."""
    return [
        {
            "instruction": "Translate 'hello' to French.",
            "input": "",
            "output": "bonjour",
        },
        {
            "instruction": "What is 2+2?",
            "input": "",
            "output": "4",
        },
        {
            "instruction": "Write a haiku about coding.",
            "input": "Topic: Python",
            "output": "Indent with soft tabs / List comprehensions sing light / Bug hides in plain sight",
        },
    ]


@pytest.fixture
def sample_alpaca_file(
    sample_alpaca_data: list[dict[str, Any]],
) -> str:
    """Write sample Alpaca data to a temp JSON file and return the path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(sample_alpaca_data, f, ensure_ascii=False)
    return f.name


@pytest.fixture
def sample_sharegpt_data() -> list[dict[str, Any]]:
    """Return a minimal ShareGPT-format dataset."""
    return [
        {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there! How can I help?"},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "What is Python?"},
                {"from": "gpt", "value": "Python is a programming language."},
            ]
        },
    ]


@pytest.fixture
def sample_sharegpt_file(
    sample_sharegpt_data: list[dict[str, Any]],
) -> str:
    """Write sample ShareGPT data to a temp JSON file and return the path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(sample_sharegpt_data, f, ensure_ascii=False)
    return f.name


@pytest.fixture
def sample_dataset_generator_data() -> list[dict[str, Any]]:
    """Return a minimal Dataset Generator format dataset."""
    return [
        {
            "status": "success",
            "conversation": [
                {"role": "user", "content": "What is machine learning?"},
                {
                    "role": "assistant",
                    "content": "Machine learning is a subset of AI.",
                },
            ],
            "summary": "ML definition",
        },
        {
            "status": "success",
            "conversation": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain NLP."},
                {
                    "role": "assistant",
                    "content": "NLP stands for Natural Language Processing.",
                },
            ],
            "summary": "NLP explanation",
        },
        {
            "status": "failed",
            "conversation": [{"role": "user", "content": "Invalid sample"}],
            "summary": "Should be filtered out",
        },
    ]


@pytest.fixture
def sample_dataset_generator_file(
    sample_dataset_generator_data: list[dict[str, Any]],
) -> str:
    """Write sample Dataset Generator data to a temp JSON file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(sample_dataset_generator_data, f, ensure_ascii=False)
    return f.name


@pytest.fixture
def temp_output_dir() -> str:
    """Create a temporary output directory and return its path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
