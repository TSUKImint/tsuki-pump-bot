"""Tests for the 6 filter modules."""

from __future__ import annotations

from pathlib import Path

from tsukibot_pump.config import (
    BundleClusterConfig,
    ConvergenceConfig,
    CtoRevivalConfig,
    CurveGraduationConfig,
    DevBlacklistConfig,
    FirstKolTouchConfig,
)
from tsukibot_pump.filters.bundle_detector import BundleDetector
from tsukibot_pump.filters.convergence import ConvergenceDetector
from tsukibot_pump.filters.cto_detector import CtoDetector
from tsukibot_pump.filters.curve_predictor import CurvePredictor
from tsukibot_pump.filters.dev_blacklist import DevBlacklist
from tsukibot_pump.filters.first_kol_touch import FirstKolTouch
from tsukibot_pump.models import BuyRecord, TokenState

# ── dev_blacklist ──────────────────────────────────────────────────────────


def _write_blacklist(path: Path, *wallets: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        for w in wallets:
            f.write(w + "\n")


def test_dev_blacklist_hard_rejects_known_rugger(tmp_path: Path) -> None:
    bl = tmp_path / "scammers.csv"
    _write_blacklist(bl, "RUGGER1", "RUGGER2")
    config = DevBlacklistConfig(
        enabled=True,
        reject_if_known_rugger=True,
        reject_if_dev_token_count_24h_gte=5,
        reject_if_dev_token_count_7d_gte=15,
        blacklist_csv_path=bl,
    )
    flt = DevBlacklist(config)
    token = TokenState(mint="M", dev_wallet="RUGGER1")
    out = flt.evaluate(token)
    assert out.hard_reject
    assert out.score == 0


def test_dev_blacklist_rejects_serial_launcher(tmp_path: Path) -> None:
    bl = tmp_path / "scammers.csv"
    bl.write_text("", encoding="utf-8")
    config = DevBlacklistConfig(
        enabled=True,
        reject_if_known_rugger=True,
        reject_if_dev_token_count_24h_gte=5,
        reject_if_dev_token_count_7d_gte=15,
        blacklist_csv_path=bl,
    )
    flt = DevBlacklist(config)
    token = TokenState(mint="M", dev_wallet="NEWDEV")
    out = flt.evaluate(token, dev_token_count_24h=6, dev_token_count_7d=10)
    assert out.hard_reject
    assert "24h" in out.reject_reason


def test_dev_blacklist_first_launcher_gets_full_score(tmp_path: Path) -> None:
    bl = tmp_path / "scammers.csv"
    bl.write_text("", encoding="utf-8")
    config = DevBlacklistConfig(
        enabled=True,
        reject_if_known_rugger=True,
        reject_if_dev_token_count_24h_gte=5,
        reject_if_dev_token_count_7d_gte=15,
        blacklist_csv_path=bl,
    )
    flt = DevBlacklist(config)
    token = TokenState(mint="M", dev_wallet="CLEANDEV")
    out = flt.evaluate(token, dev_token_count_24h=0, dev_token_count_7d=0)
    assert out.score == 100.0
    assert not out.hard_reject


def test_dev_blacklist_disabled_returns_neutral(tmp_path: Path) -> None:
    bl = tmp_path / "scammers.csv"
    bl.write_text("", encoding="utf-8")
    config = DevBlacklistConfig(
        enabled=False,
        reject_if_known_rugger=True,
        reject_if_dev_token_count_24h_gte=5,
        reject_if_dev_token_count_7d_gte=15,
        blacklist_csv_path=bl,
    )
    flt = DevBlacklist(config)
    token = TokenState(mint="M", dev_wallet="WHATEVER")
    out = flt.evaluate(token)
    assert out.score == 50.0


# ── bundle_detector ────────────────────────────────────────────────────────


def _bundle_config() -> BundleClusterConfig:
    return BundleClusterConfig(
        enabled=True,
        reject_cluster_concentration_gte=0.30,
        first_n_buyers=10,
        bundle_window_slots=1,
    )


def test_bundle_detector_neutral_when_too_few_buys() -> None:
    flt = BundleDetector(_bundle_config())
    token = TokenState(mint="M")
    token.buys.append(BuyRecord("W", 0.1, 100, 1, 1, "s"))
    out = flt.evaluate(token)
    assert out.score == 50.0


def test_bundle_detector_rewards_organic_distribution() -> None:
    flt = BundleDetector(_bundle_config())
    token = TokenState(mint="M")
    # 10 distinct wallets, distinct slots, similar amounts → low concentration.
    for i in range(10):
        token.buys.append(
            BuyRecord(
                f"W{i}", 0.1 + i * 0.001, 100, slot=i + 1, block_time_unix=i, signature=f"s{i}"
            )
        )
    out = flt.evaluate(token)
    assert out.score == 100.0
    assert not out.hard_reject


def test_bundle_detector_rejects_concentrated_cluster() -> None:
    flt = BundleDetector(_bundle_config())
    token = TokenState(mint="M")
    # Wallet W0 gets 50% of supply, W1..W9 share the rest.
    token.buys.append(BuyRecord("W0", 5.0, 5_000_000, slot=1, block_time_unix=0, signature="s0"))
    for i in range(1, 10):
        token.buys.append(
            BuyRecord(f"W{i}", 0.5, 500_000, slot=i + 1, block_time_unix=i, signature=f"s{i}")
        )
    out = flt.evaluate(token)
    assert out.hard_reject
    assert "concentration" in out.reject_reason or "cluster" in out.reject_reason


def test_bundle_detector_detects_same_amount_snipe() -> None:
    flt = BundleDetector(_bundle_config())
    token = TokenState(mint="M")
    # 5 different wallets all buying identical 0.5 SOL — sniper pattern.
    for i in range(5):
        token.buys.append(
            BuyRecord(f"S{i}", 0.5, 100, slot=i + 1, block_time_unix=i, signature=f"s{i}")
        )
    for i in range(5, 10):
        token.buys.append(
            BuyRecord(f"O{i}", 0.1, 100, slot=i + 1, block_time_unix=i, signature=f"s{i}")
        )
    out = flt.evaluate(token, wallet_funder_lookup={f"S{i}": "FUNDER" for i in range(5)})
    # Stealth signals + concentration both fire.
    assert out.hard_reject or out.score <= 20.0


# ── first_kol_touch ────────────────────────────────────────────────────────


def _write_kol(path: Path, *entries: tuple[str, str, float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for wallet, label, score in entries:
            f.write(f"{wallet},{label},{score}\n")


def test_first_kol_touch_no_match() -> None:
    kol = Path("/tmp") / "_kol_test_1.csv"
    _write_kol(kol, ("KOL_A", "alpha", 90))
    config = FirstKolTouchConfig(
        enabled=True,
        first_n_buyers=10,
        kol_csv_path=kol,
        score_per_touch=25.0,
        max_kol_touches=3,
    )
    flt = FirstKolTouch(config)
    token = TokenState(mint="M")
    token.buys = [
        BuyRecord(f"W{i}", 0.1, 100, slot=i, block_time_unix=i, signature=f"s{i}") for i in range(5)
    ]
    out = flt.evaluate(token)
    assert out.score == 30.0


def test_first_kol_touch_with_kol_at_position_2(tmp_path: Path) -> None:
    kol = tmp_path / "kol.csv"
    _write_kol(kol, ("KOL_A", "alpha", 100))
    config = FirstKolTouchConfig(
        enabled=True,
        first_n_buyers=10,
        kol_csv_path=kol,
        score_per_touch=25.0,
        max_kol_touches=3,
    )
    flt = FirstKolTouch(config)
    token = TokenState(mint="M")
    token.buys = [
        BuyRecord("W0", 0.1, 100, slot=1, block_time_unix=1, signature="s0"),
        BuyRecord("KOL_A", 0.1, 100, slot=2, block_time_unix=2, signature="s1"),
        BuyRecord("W2", 0.1, 100, slot=3, block_time_unix=3, signature="s2"),
    ]
    out = flt.evaluate(token)
    # 1 touch * 25 = 25, +20% conviction bonus = 30 → scaled by avg KOL score 100/100 → 30.
    assert out.score >= 25.0


# ── convergence ────────────────────────────────────────────────────────────


def _convergence_config() -> ConvergenceConfig:
    return ConvergenceConfig(
        enabled=True,
        window_seconds=600,
        min_distinct_kols=2,
        score_at_minimum=60.0,
        score_per_extra_kol=15.0,
    )


def test_convergence_two_kols_in_window(tmp_path: Path) -> None:
    kol = tmp_path / "kol.csv"
    _write_kol(kol, ("K1", "k1", 100), ("K2", "k2", 100))
    flt = ConvergenceDetector(_convergence_config(), kol_csv_path=kol)
    token = TokenState(mint="M")
    token.buys = [
        BuyRecord("K1", 0.1, 100, slot=1, block_time_unix=1000, signature="s"),
        BuyRecord("K2", 0.1, 100, slot=2, block_time_unix=1100, signature="s2"),
    ]
    out = flt.evaluate(token)
    assert out.score >= 60.0


def test_convergence_too_far_apart(tmp_path: Path) -> None:
    kol = tmp_path / "kol.csv"
    _write_kol(kol, ("K1", "k1", 100), ("K2", "k2", 100))
    flt = ConvergenceDetector(_convergence_config(), kol_csv_path=kol)
    token = TokenState(mint="M")
    token.buys = [
        BuyRecord("K1", 0.1, 100, slot=1, block_time_unix=1000, signature="s"),
        BuyRecord("K2", 0.1, 100, slot=2, block_time_unix=2000, signature="s2"),  # 1000s later
    ]
    out = flt.evaluate(token)
    # Outside 600s window — should NOT count as convergence.
    assert out.score == 40.0


# ── curve_predictor ────────────────────────────────────────────────────────


def _curve_config() -> CurveGraduationConfig:
    return CurveGraduationConfig(
        enabled=True,
        enter_after_sol_in_curve_gte=65.0,
        min_velocity_sol_per_min=0.5,
        min_distinct_buyers_60s=8,
    )


def test_curve_below_threshold_returns_low() -> None:
    flt = CurvePredictor(_curve_config())
    token = TokenState(mint="M")
    token.sol_in_curve = 30.0
    token.last_sol_velocity_sol_per_min = 1.0
    token.distinct_buyers_60s = 10
    out = flt.evaluate(token)
    assert out.score == 40.0


def test_curve_all_gates_passed() -> None:
    flt = CurvePredictor(_curve_config())
    token = TokenState(mint="M")
    token.sol_in_curve = 75.0
    token.last_sol_velocity_sol_per_min = 1.5
    token.distinct_buyers_60s = 12
    out = flt.evaluate(token)
    assert out.score >= 60.0


def test_curve_complete_returns_neutral() -> None:
    flt = CurvePredictor(_curve_config())
    token = TokenState(mint="M")
    token.curve_complete = True
    out = flt.evaluate(token)
    assert out.score == 50.0


# ── cto_detector ───────────────────────────────────────────────────────────


def _cto_config() -> CtoRevivalConfig:
    return CtoRevivalConfig(
        enabled=True,
        dev_silent_min_days=2,
        min_unique_buyers_24h=30,
        min_days_since_launch=3,
    )


def test_cto_too_fresh() -> None:
    flt = CtoDetector(_cto_config())
    token = TokenState(mint="M")
    now = 1_700_000_000
    token.created_at_unix = now - 86400  # 1 day ago
    out = flt.evaluate(token, now_unix=now, unique_buyers_24h=100)
    assert out.score == 30.0


def test_cto_all_criteria_met() -> None:
    flt = CtoDetector(_cto_config())
    token = TokenState(mint="M")
    now = 1_700_000_000
    token.created_at_unix = now - 7 * 86400  # 7 days old
    token.dev_activity.last_seen_unix = now - 3 * 86400  # dev silent 3 days
    out = flt.evaluate(token, now_unix=now, unique_buyers_24h=50)
    assert out.score == 100.0


def test_cto_partial_criteria() -> None:
    flt = CtoDetector(_cto_config())
    token = TokenState(mint="M")
    now = 1_700_000_000
    token.created_at_unix = now - 7 * 86400
    token.dev_activity.last_seen_unix = now - 12 * 3600  # dev active 12h ago
    out = flt.evaluate(token, now_unix=now, unique_buyers_24h=50)
    # dev not silent, but other criteria met → 3/4
    assert out.score == 60.0
