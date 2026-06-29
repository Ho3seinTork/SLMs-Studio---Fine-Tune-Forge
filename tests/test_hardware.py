"""Tests for the hardware detection and profile recommendation module."""

import sys

import pytest

from app import hardware


def test_detect_hardware_returns_expected_keys():
    """detect_hardware() should return a dict with all expected keys."""
    hw = hardware.detect_hardware()
    expected_keys = {
        "os",
        "os_version",
        "python_version",
        "cpu_name",
        "cpu_cores_logical",
        "ram_total_gb",
        "ram_available_gb",
        "gpu_available",
        "gpu_name",
        "gpu_vram_total_gb",
        "cuda_version",
        "torch_installed",
    }
    assert expected_keys.issubset(hw.keys())


def test_detect_hardware_has_valid_types():
    """detect_hardware() values should have reasonable types."""
    hw = hardware.detect_hardware()

    assert isinstance(hw["os"], str)
    assert isinstance(hw["cpu_cores_logical"], int)
    assert hw["cpu_cores_logical"] >= 1
    assert isinstance(hw["gpu_available"], bool)
    assert isinstance(hw["torch_installed"], bool)

    if hw["ram_total_gb"] is not None:
        assert isinstance(hw["ram_total_gb"], float)
        assert hw["ram_total_gb"] > 0


def test_bytes_to_gb():
    """_bytes_to_gb should correctly convert bytes to GB."""
    assert hardware._bytes_to_gb(1024**3) == 1.0
    assert hardware._bytes_to_gb(2 * 1024**3) == 2.0
    assert hardware._bytes_to_gb(512 * 1024**2) == 0.5


@pytest.mark.parametrize(
    "profile_key",
    ["laptop_cpu", "laptop_gpu_small", "server_gpu"],
)
def test_all_profiles_have_required_keys(profile_key: str):
    """Every profile should contain all expected hyperparameter keys."""
    profile = hardware.PROFILES[profile_key]
    required = {
        "label",
        "max_model_params_b",
        "use_4bit",
        "precision",
        "lora_r",
        "lora_alpha",
        "per_device_batch_size",
        "grad_accum_steps",
        "max_seq_len",
        "gradient_checkpointing",
        "warning",
    }
    assert required.issubset(profile.keys())


def test_recommend_profile_without_gpu():
    """Without a GPU, should recommend laptop_cpu."""
    hw = {
        "gpu_available": False,
        "gpu_vram_total_gb": 0,
    }
    profile = hardware.recommend_profile(hw)
    assert profile["key"] == "laptop_cpu"
    assert "warning" in profile
    assert profile["use_4bit"] is False


def test_recommend_profile_with_small_gpu():
    """With a small GPU (less than 16 GB VRAM), should recommend laptop_gpu_small."""
    hw = {
        "gpu_available": True,
        "gpu_vram_total_gb": 6,
    }
    profile = hardware.recommend_profile(hw)
    assert profile["key"] == "laptop_gpu_small"


def test_recommend_profile_with_large_gpu():
    """With a large GPU (≥16 GB VRAM), should recommend server_gpu."""
    hw = {
        "gpu_available": True,
        "gpu_vram_total_gb": 24,
    }
    profile = hardware.recommend_profile(hw)
    assert profile["key"] == "server_gpu"


def test_recommend_profile_boundary_16gb():
    """At exactly 16 GB VRAM, should recommend server_gpu."""
    hw = {
        "gpu_available": True,
        "gpu_vram_total_gb": 16,
    }
    profile = hardware.recommend_profile(hw)
    assert profile["key"] == "server_gpu"


def test_recommend_profile_includes_hardware():
    """The returned profile should embed the hardware info."""
    hw = hardware.detect_hardware()
    profile = hardware.recommend_profile(hw)
    assert profile["hardware"] == hw
