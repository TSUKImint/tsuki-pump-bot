"""Pytest fixtures shared across the suite."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import tsukibot_pump.config as config_mod


@pytest.fixture
def example_config_dict() -> dict:
    """Load the committed example config as a plain dict so tests can mutate it."""
    path = Path(__file__).resolve().parent.parent / "config.example.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def example_config(example_config_dict: dict) -> config_mod.Config:
    return config_mod.Config.model_validate(example_config_dict)


@pytest.fixture
def tmp_state_dir(tmp_path: Path) -> Path:
    d = tmp_path / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d
