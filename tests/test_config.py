"""Config loader + validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tsukibot_pump.config import Config, Settings, load_config


def test_example_config_is_valid(example_config: Config) -> None:
    assert example_config.bankroll.total_sol > 0
    assert example_config.scoring.enter_threshold >= 0
    # Weights must sum to 1.0
    w = example_config.scoring.weights
    total = (
        w.dev_blacklist
        + w.bundle_cluster
        + w.first_kol_touch
        + w.convergence
        + w.curve_graduation
        + w.cto_revival
    )
    assert abs(total - 1.0) < 1e-9


def test_load_config_accepts_str_path(tmp_path: Path, example_config_dict: dict) -> None:
    """Regression: load_config must accept both str and Path."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(example_config_dict), encoding="utf-8")
    cfg = load_config(str(cfg_path))
    assert isinstance(cfg, Config)


def test_load_config_accepts_pathlib_path(tmp_path: Path, example_config_dict: dict) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(example_config_dict), encoding="utf-8")
    cfg = load_config(cfg_path)
    assert isinstance(cfg, Config)


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_load_config_non_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(bad)


def test_scoring_weights_must_sum_to_one(example_config_dict: dict) -> None:
    example_config_dict["scoring"]["weights"]["dev_blacklist"] = 0.5  # break it
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        Config.model_validate(example_config_dict)


def test_exit_ladder_must_be_monotonic(example_config_dict: dict) -> None:
    example_config_dict["exits"]["ladder"] = [
        {"roi": 3.0, "sell_fraction": 0.3},
        {"roi": 1.0, "sell_fraction": 0.5},  # backwards
    ]
    with pytest.raises(ValueError, match="ascending roi"):
        Config.model_validate(example_config_dict)


def test_exit_ladder_fractions_cannot_exceed_one(example_config_dict: dict) -> None:
    example_config_dict["exits"]["ladder"] = [
        {"roi": 1.0, "sell_fraction": 0.7},
        {"roi": 3.0, "sell_fraction": 0.7},  # total 1.4
    ]
    with pytest.raises(ValueError, match="sell_fractions sum"):
        Config.model_validate(example_config_dict)


def test_settings_mode_default() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.tsuki_pump_mode == "paper"
    assert s.is_live_chain is True
    assert s.places_real_orders is False
    assert s.risks_real_money is False


def test_settings_paper_mock_skips_live_chain() -> None:
    s = Settings(_env_file=None, tsuki_pump_mode="paper-mock")  # type: ignore[call-arg]
    assert s.is_live_chain is False
    assert s.places_real_orders is False
    assert s.risks_real_money is False


def test_settings_mainnet_flags() -> None:
    s = Settings(_env_file=None, tsuki_pump_mode="mainnet")  # type: ignore[call-arg]
    assert s.is_live_chain is True
    assert s.places_real_orders is True
    assert s.risks_real_money is True
    assert s.effective_rpc_url == s.solana_rpc_url


def test_settings_devnet_uses_devnet_url() -> None:
    s = Settings(_env_file=None, tsuki_pump_mode="devnet")  # type: ignore[call-arg]
    assert s.effective_rpc_url == s.solana_devnet_rpc_url
