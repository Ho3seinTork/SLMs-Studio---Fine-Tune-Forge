"""Tests for the dataset converter module."""

import json
from pathlib import Path

import pytest

from app import dataset_converter


class TestLoadAndNormalize:
    """Tests for load_and_normalize() format detection and conversion."""

    def test_alpaca_format_detection(self, sample_alpaca_file: str):
        """Should detect Alpaca format and convert correctly."""
        examples, fmt = dataset_converter.load_and_normalize(
            sample_alpaca_file
        )
        assert fmt == "alpaca"
        assert len(examples) == 3
        for ex in examples:
            assert "messages" in ex
            assert len(ex["messages"]) == 2
            assert ex["messages"][0]["role"] == "user"
            assert ex["messages"][1]["role"] == "assistant"

    def test_alpaca_with_input_field(self):
        """Alpaca items with an 'input' field should be concatenated."""
        import tempfile

        data = [
            {"instruction": "Q", "input": "Context", "output": "A"}
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            path = f.name

        examples, fmt = dataset_converter.load_and_normalize(path)
        assert fmt == "alpaca"
        user_content = examples[0]["messages"][0]["content"]
        assert "Context" in user_content
        assert "Q" in user_content

    def test_sharegpt_format_detection(self, sample_sharegpt_file: str):
        """Should detect ShareGPT format and convert correctly."""
        examples, fmt = dataset_converter.load_and_normalize(
            sample_sharegpt_file
        )
        assert fmt == "sharegpt"
        assert len(examples) == 2
        for ex in examples:
            assert len(ex["messages"]) == 2

    def test_dataset_generator_format(self, sample_dataset_generator_file: str):
        """Should detect Dataset Generator format and filter failures."""
        examples, fmt = dataset_converter.load_and_normalize(
            sample_dataset_generator_file
        )
        assert fmt == "dataset_generator"
        # Only 2 success samples (the failed one is filtered out)
        assert len(examples) == 2

    def test_empty_list_raises(self):
        """An empty JSON array should raise ValueError."""
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump([], f)
            path = f.name

        with pytest.raises(ValueError, match="non-empty"):
            dataset_converter.load_and_normalize(path)

    def test_unrecognized_format_raises(self):
        """An unrecognized format should raise ValueError."""
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump([{"unknown_field": "value"}], f)
            path = f.name

        with pytest.raises(ValueError, match="Unrecognized"):
            dataset_converter.load_and_normalize(path)

    def test_not_a_list_raises(self):
        """A JSON object (not array) should raise ValueError."""
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"key": "value"}, f)
            path = f.name

        with pytest.raises(ValueError, match="non-empty JSON array"):
            dataset_converter.load_and_normalize(path)


class TestSplitAndWrite:
    """Tests for split_and_write()."""

    def test_basic_split(self, sample_alpaca_file: str, temp_output_dir: str):
        """Should create train.jsonl with correct counts.

        Note: with fewer than 10 examples, eval split is skipped entirely
        (all data goes to training). This test uses a 3-example dataset
        so we expect no eval file.
        """
        examples, _ = dataset_converter.load_and_normalize(
            sample_alpaca_file
        )
        result = dataset_converter.split_and_write(
            examples, temp_output_dir, eval_ratio=0.3, seed=42
        )

        train_path = Path(result["train_file"])
        assert train_path.exists()

        # With 3 samples (<10), eval is skipped — all data goes to train
        assert result["n_train"] == 3
        assert result["n_eval"] == 0
        assert result["eval_file"] is None

        # Verify JSONL format
        train_lines = train_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(train_lines) == 3
        for line in train_lines:
            obj = json.loads(line)
            assert "messages" in obj

    def test_small_dataset_no_eval(self, temp_output_dir: str):
        """Datasets with fewer than 10 examples should have no eval split."""
        import tempfile

        # Create a dataset with only 5 examples
        data = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(5)]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            path = f.name

        examples, _ = dataset_converter.load_and_normalize(path)
        result = dataset_converter.split_and_write(examples, temp_output_dir)

        assert result["n_eval"] == 0
        assert result["eval_file"] is None
        assert result["n_train"] == 5

    def test_reproducible_split(self, temp_output_dir: str):
        """Same seed should produce the same split order."""
        import tempfile

        data = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(20)]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            path = f.name

        examples, _ = dataset_converter.load_and_normalize(path)

        r1 = dataset_converter.split_and_write(
            examples, temp_output_dir + "/1", seed=42
        )
        r2 = dataset_converter.split_and_write(
            examples, temp_output_dir + "/2", seed=42
        )

        assert r1["n_train"] == r2["n_train"]
        assert r1["n_eval"] == r2["n_eval"]


class TestPreview:
    """Tests for preview()."""

    def test_preview_returns_format_and_sample(
        self, sample_alpaca_file: str
    ):
        """Preview should return format, total count, and sample items."""
        result = dataset_converter.preview(sample_alpaca_file, n=2)
        assert result["format_detected"] == "alpaca"
        assert result["total_examples"] == 3
        assert len(result["sample"]) == 2


class TestPrepareDataset:
    """Tests for the full prepare_dataset() pipeline."""

    def test_prepare_dataset_returns_all_stats(
        self, sample_alpaca_file: str, temp_output_dir: str
    ):
        """prepare_dataset should return format, counts, and file paths."""
        result = dataset_converter.prepare_dataset(
            sample_alpaca_file, temp_output_dir
        )
        expected_keys = {
            "train_file",
            "eval_file",
            "n_train",
            "n_eval",
            "format_detected",
            "total_examples",
        }
        assert expected_keys.issubset(result.keys())
        assert result["format_detected"] == "alpaca"
        assert result["total_examples"] == 3
