"""Tests for the v0.3 aggressive paper-trader profile + early-conviction lane.

These pieces are opt-in. When `aggressive_paper.enabled=False`, the config
is unchanged. When `True`, the named knobs override fields elsewhere in the
config so the orchestrator can stay unchanged.

The early-conviction lane requires all of:
  - on-chain creator field is known
  - creator has >= min_prior_graduations
  - >= min_kol_touches distinct tracked KOLs bought within window_seconds
    of CREATE.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from tsukibot_pump.config import load_config
from tsukibot_pump.models import BuyRecord, TokenState
from tsukibot_pump.strategy_profile import (
    apply_aggressive_overrides,
    count_kol_touches_in_window,
    early_conviction_cap_sol,
    is_early_conviction_signal,
)


def _load_with_overrides(**overrides: dict[str, object]) -> object:
    """Load the example config and apply per-section overrides."""
    raw = yaml.safe_load(Path("config.example.yaml").read_text())
    for section, kvs in overrides.items():
        if isinstance(kvs, dict):
            for k, v in kvs.items():
                if "." in k:
                    head, tail = k.split(".", 1)
                    raw[section].setdefault(head, {})[tail] = v
                else:
                    raw[section][k] = v
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        yaml.safe_dump(raw, tmp)
        tmp_path = tmp.name
    return load_config(tmp_path)


def test_aggressive_profile_disabled_leaves_config_unchanged() -> None:
    c = _load_with_overrides()  # default: aggressive disabled
    c2, profile = apply_aggressive_overrides(c)
    assert profile.aggressive_enabled is False
    assert c2.scoring.enter_threshold == c.scoring.enter_threshold
    assert c2.sizing.fraction_of_kelly == c.sizing.fraction_of_kelly
    assert (
        c2.filters.curve_graduation.enter_after_sol_in_curve_gte
        == c.filters.curve_graduation.enter_after_sol_in_curve_gte
    )


def test_aggressive_profile_enabled_applies_all_overrides() -> None:
    c = _load_with_overrides(scoring={"aggressive_paper.enabled": True})
    c2, profile = apply_aggressive_overrides(c)
    assert profile.aggressive_enabled is True
    # Defaults from the YAML:
    assert c2.scoring.enter_threshold == 40
    assert c2.sizing.fraction_of_kelly == 0.50
    assert c2.sizing.min_expected_roi == 0.10
    assert c2.bankroll.single_token_cap_fraction == 0.10
    assert c2.filters.curve_graduation.enter_after_sol_in_curve_gte == 25.0
    assert c2.filters.curve_graduation.min_velocity_sol_per_min == 0.2
    assert c2.watch.http_poll_interval_seconds == 1.5
    assert profile.scoring_cycle_seconds == 2.5


def test_aggressive_profile_preserves_unrelated_fields() -> None:
    """Overriding aggressive knobs must not change the curve graduation
    `min_distinct_buyers_60s` (which the profile doesn't touch)."""
    c = _load_with_overrides(scoring={"aggressive_paper.enabled": True})
    c2, _ = apply_aggressive_overrides(c)
    assert (
        c2.filters.curve_graduation.min_distinct_buyers_60s
        == c.filters.curve_graduation.min_distinct_buyers_60s
    )


# ── Early-conviction lane ─────────────────────────────────────────────────


def _token_with_buys(
    *,
    creator: str | None = "CREATOR",
    creator_prior_graduations: int = 0,
    created_at_unix: int = 1_700_000_000,
    buys: list[tuple[str, int]] | None = None,
) -> TokenState:
    t = TokenState(
        mint="MINT",
        creator=creator,
        creator_prior_graduations=creator_prior_graduations,
        created_at_unix=created_at_unix,
    )
    if buys:
        t.buys = [
            BuyRecord(
                wallet=wallet,
                sol_spent=0.1,
                token_units_received=1_000.0,
                slot=1,
                block_time_unix=block_time,
                signature=f"sig_{i}",
            )
            for i, (wallet, block_time) in enumerate(buys)
        ]
    return t


def test_count_kol_touches_in_window_no_kols_returns_zero() -> None:
    t = _token_with_buys(buys=[("WALLET_A", 1_700_000_010)])
    n = count_kol_touches_in_window(t, kol_wallets=set(), window_seconds=30)
    assert n == 0


def test_count_kol_touches_in_window_returns_distinct_count() -> None:
    t = _token_with_buys(
        buys=[
            ("KOL_A", 1_700_000_005),
            ("KOL_A", 1_700_000_010),  # duplicate same KOL
            ("KOL_B", 1_700_000_020),
            ("NOT_KOL", 1_700_000_025),
        ]
    )
    n = count_kol_touches_in_window(t, kol_wallets={"KOL_A", "KOL_B"}, window_seconds=30)
    assert n == 2  # distinct, not 3


def test_count_kol_touches_excludes_buys_outside_window() -> None:
    t = _token_with_buys(
        buys=[
            ("KOL_A", 1_700_000_005),
            ("KOL_B", 1_700_000_100),  # outside 30s window
        ]
    )
    n = count_kol_touches_in_window(t, kol_wallets={"KOL_A", "KOL_B"}, window_seconds=30)
    assert n == 1


def test_early_conviction_signal_requires_creator_field() -> None:
    """No creator field ⇒ never trigger (we can't verify graduation history)."""
    t = _token_with_buys(
        creator=None,
        buys=[("KOL_A", 1_700_000_005), ("KOL_B", 1_700_000_010)],
    )
    assert (
        is_early_conviction_signal(
            t,
            kol_wallets={"KOL_A", "KOL_B"},
            window_seconds=30,
            min_kol_touches=2,
            min_prior_graduations=1,
        )
        is False
    )


def test_early_conviction_signal_requires_prior_graduations() -> None:
    """Creator with zero prior graduations ⇒ never trigger."""
    t = _token_with_buys(
        creator_prior_graduations=0,
        buys=[("KOL_A", 1_700_000_005), ("KOL_B", 1_700_000_010)],
    )
    assert (
        is_early_conviction_signal(
            t,
            kol_wallets={"KOL_A", "KOL_B"},
            window_seconds=30,
            min_kol_touches=2,
            min_prior_graduations=1,
        )
        is False
    )


def test_early_conviction_signal_requires_enough_kol_touches() -> None:
    """Aligned creator, but only 1 KOL touch in window ⇒ no trigger."""
    t = _token_with_buys(
        creator_prior_graduations=2,
        buys=[("KOL_A", 1_700_000_005)],
    )
    assert (
        is_early_conviction_signal(
            t,
            kol_wallets={"KOL_A", "KOL_B"},
            window_seconds=30,
            min_kol_touches=2,
            min_prior_graduations=1,
        )
        is False
    )


def test_early_conviction_signal_fires_when_all_gates_met() -> None:
    """The happy path: aligned creator + 2 KOL touches in window."""
    t = _token_with_buys(
        creator_prior_graduations=2,
        buys=[("KOL_A", 1_700_000_005), ("KOL_B", 1_700_000_010)],
    )
    assert (
        is_early_conviction_signal(
            t,
            kol_wallets={"KOL_A", "KOL_B"},
            window_seconds=30,
            min_kol_touches=2,
            min_prior_graduations=1,
        )
        is True
    )


def test_early_conviction_cap_sol_is_min_of_normal_and_lane_caps() -> None:
    """The early-conviction cap is never larger than the normal cap."""
    c = _load_with_overrides(
        scoring={
            "early_conviction.enabled": True,
            "early_conviction.max_single_token_cap_fraction": 0.03,
        },
    )
    bankroll = c.bankroll.total_sol
    cap = early_conviction_cap_sol(c, bankroll)
    # Normal cap: 0.05 * 10 = 0.5; lane cap: 0.03 * 10 = 0.3 → min = 0.3
    assert cap == 0.3
