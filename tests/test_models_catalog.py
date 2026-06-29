"""Tests for the models catalog module."""

from app import models_catalog


def test_models_list_is_not_empty():
    """The model catalog should contain at least one model."""
    assert len(models_catalog.MODELS) > 0


def test_each_model_has_required_fields():
    """Every model entry should have all required fields."""
    required = {
        "id",
        "label",
        "params_b",
        "gated",
        "min_ram_gb_cpu",
        "min_vram_gb_qlora",
        "notes",
    }
    for m in models_catalog.MODELS:
        assert required.issubset(m.keys()), f"Missing fields in {m['id']}"
        assert isinstance(m["id"], str)
        assert isinstance(m["params_b"], (int, float))
        assert isinstance(m["gated"], bool)


def test_models_for_profile_filters_by_params():
    """models_for_profile returns all models but flags unfit ones as False."""
    profile = {"max_model_params_b": 1.0, "key": "test"}
    result = models_catalog.models_for_profile(profile)
    for m in result:
        if m["params_b"] > 1.0:
            # Models exceeding the budget should be marked as unfit
            assert m["fits_current_hardware"] is False
        else:
            assert m["fits_current_hardware"] is True


def test_models_for_profile_adds_fits_flag():
    """Each returned model should have a fits_current_hardware boolean."""
    profile = {"max_model_params_b": 5.0, "key": "test"}
    result = models_catalog.models_for_profile(profile)
    for m in result:
        assert "fits_current_hardware" in m
        assert isinstance(m["fits_current_hardware"], bool)


def test_models_for_profile_all_fit_with_large_budget():
    """With a large budget, all models should fit."""
    profile = {"max_model_params_b": 100.0, "key": "test"}
    result = models_catalog.models_for_profile(profile)
    assert all(m["fits_current_hardware"] for m in result)


def test_models_for_profile_none_fit_with_tiny_budget():
    """With a tiny budget, no models should fit."""
    profile = {"max_model_params_b": 0.1, "key": "test"}
    result = models_catalog.models_for_profile(profile)
    assert not any(m["fits_current_hardware"] for m in result)
