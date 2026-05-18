"""Smoke tests for the CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tsukibot_pump.__main__ import _parse_args, main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        _parse_args(["--version"])
    assert exc.value.code == 0


def test_paper_trader_toggle() -> None:
    args = _parse_args(["--no-paper-trader"])
    assert args.paper_trader is False
    args = _parse_args([])
    assert args.paper_trader is True


def test_devnet_without_ack_refused(
    tmp_path: Path,
    example_config_dict: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live modes must refuse to start without the explicit ack flag."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(example_config_dict), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TSUKI_PUMP_MODE", raising=False)
    exit_code = main(["--mode", "devnet", "--config", str(cfg_path), "--no-dashboard", "--once"])
    assert exit_code == 2


def test_paper_mock_once_runs_and_exits(
    tmp_path: Path,
    example_config_dict: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--mode paper-mock --once should complete without a network call."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(example_config_dict), encoding="utf-8")
    # Point seed CSVs at non-existent paths so we don't read the demo files.
    example_config_dict["filters"]["dev_blacklist"]["blacklist_csv_path"] = str(tmp_path / "x.csv")
    example_config_dict["filters"]["first_kol_touch"]["kol_csv_path"] = str(tmp_path / "y.csv")
    example_config_dict["event_store"]["sqlite_path"] = str(tmp_path / "e.db")
    example_config_dict["event_store"]["jsonl_audit_path"] = str(tmp_path / "a.jsonl")
    cfg_path.write_text(yaml.safe_dump(example_config_dict), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TSUKI_PUMP_MODE", raising=False)
    exit_code = main(
        ["--mode", "paper-mock", "--config", str(cfg_path), "--no-dashboard", "--once"]
    )
    assert exit_code == 0
