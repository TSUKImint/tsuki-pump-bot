"""PaperExecutor tests."""

from __future__ import annotations

import random

import pytest

from tsukibot_pump.config import PaperRealismConfig
from tsukibot_pump.execution.paper_executor import (
    PaperExecutor,
    PaperFillFailedError,
    _advance_curve_by_sol,
    _sample_lognormal,
)
from tsukibot_pump.solana.bonding_curve import (
    DEFAULT_VIRTUAL_SOL_RESERVES,
    DEFAULT_VIRTUAL_TOKEN_RESERVES,
    LAMPORTS_PER_SOL,
    BondingCurveState,
)


def _fresh_curve() -> BondingCurveState:
    return BondingCurveState(
        virtual_sol_reserves=DEFAULT_VIRTUAL_SOL_RESERVES,
        virtual_token_reserves=DEFAULT_VIRTUAL_TOKEN_RESERVES,
        real_sol_reserves=0,
        real_token_reserves=793_100_000 * 10**6,
        complete=False,
    )


def test_buy_applies_slippage_upward() -> None:
    no_slip = PaperExecutor(slippage_bps=0.0)
    with_slip = PaperExecutor(slippage_bps=100.0)  # 1%
    curve = _fresh_curve()
    f1 = no_slip.buy("M", 100_000, curve)
    f2 = with_slip.buy("M", 100_000, curve)
    assert f2.notional_sol > f1.notional_sol
    assert f2.fill_price_sol_per_token > f1.fill_price_sol_per_token
    assert f1.slippage_bps_applied == 0.0
    assert f2.slippage_bps_applied == 100.0


def test_sell_applies_slippage_downward() -> None:
    no_slip = PaperExecutor(slippage_bps=0.0)
    with_slip = PaperExecutor(slippage_bps=100.0)
    curve = BondingCurveState(
        virtual_sol_reserves=DEFAULT_VIRTUAL_SOL_RESERVES + 10 * 1_000_000_000,
        virtual_token_reserves=DEFAULT_VIRTUAL_TOKEN_RESERVES - 10_000_000_000,
        real_sol_reserves=10 * 1_000_000_000,
        real_token_reserves=783_100_000 * 10**6,
        complete=False,
    )
    f1 = no_slip.sell("M", 10_000, curve)
    f2 = with_slip.sell("M", 10_000, curve)
    assert f2.notional_sol < f1.notional_sol


def test_negative_slippage_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PaperExecutor(slippage_bps=-1.0)


def test_buy_zero_units_rejected() -> None:
    exe = PaperExecutor(slippage_bps=0.0)
    with pytest.raises(ValueError, match="positive"):
        exe.buy("M", 0.0, _fresh_curve())


def test_buy_more_than_curve_rejected() -> None:
    exe = PaperExecutor(slippage_bps=0.0)
    huge_curve = _fresh_curve()
    huge_units = (huge_curve.virtual_token_reserves / 10**6) - 1
    with pytest.raises(ValueError, match="insufficient depth"):
        # Asking for everything → curve math returns inf → executor rejects.
        exe.buy("M", huge_units * 2, huge_curve)


def test_paper_fill_marked_paper() -> None:
    exe = PaperExecutor(slippage_bps=50.0)
    fill = exe.buy("MINT_X", 1000, _fresh_curve())
    assert fill.paper is True
    assert fill.side == "BUY"
    assert fill.mint == "MINT_X"
    assert fill.fill_units == 1000


# ── PR #C: paper-realism layer ────────────────────────────────────────────


def _realism(**overrides: object) -> PaperRealismConfig:
    base: dict[str, object] = {
        "enabled": True,
        "pump_fee_bps": 100.0,
        "priority_fee_lamports_p50": 50_000,
        "priority_fee_lamports_p99": 300_000,
        "end_to_end_latency_ms_p50": 1_500.0,
        "end_to_end_latency_ms_p99": 5_000.0,
        "buy_fail_prob": 0.0,
        "sell_fail_prob": 0.0,
        "rng_seed": 1,
    }
    base.update(overrides)
    return PaperRealismConfig(**base)  # type: ignore[arg-type]


def test_realism_disabled_matches_legacy_path() -> None:
    """When realism is off the executor must be bit-identical to v0.2 paper."""
    exe = PaperExecutor(slippage_bps=100.0)  # no realism arg
    fill = exe.buy("M", 50_000, _fresh_curve())
    assert fill.pump_fee_sol == 0.0
    assert fill.priority_fee_sol == 0.0
    assert fill.latency_ms == 0.0
    assert fill.drift_sol_absorbed == 0.0


def test_realism_adds_pump_fee_to_buy_notional() -> None:
    bare = PaperExecutor(slippage_bps=100.0)
    realistic = PaperExecutor(slippage_bps=100.0, realism=_realism())
    curve = _fresh_curve()
    bare_fill = bare.buy("M", 50_000, curve)
    real_fill = realistic.buy("M", 50_000, curve)
    # 1% pump fee + priority fee → notional must be strictly higher.
    assert real_fill.notional_sol > bare_fill.notional_sol
    assert real_fill.pump_fee_sol > 0.0
    # Pump fee should be ~1% of the slippage-adjusted cost.
    expected_pump_fee = bare_fill.notional_sol * 0.01
    assert abs(real_fill.pump_fee_sol - expected_pump_fee) / expected_pump_fee < 0.05


def test_realism_deducts_pump_fee_from_sell_proceeds() -> None:
    funded_curve = BondingCurveState(
        virtual_sol_reserves=DEFAULT_VIRTUAL_SOL_RESERVES + 10 * LAMPORTS_PER_SOL,
        virtual_token_reserves=DEFAULT_VIRTUAL_TOKEN_RESERVES - 10_000_000_000,
        real_sol_reserves=10 * LAMPORTS_PER_SOL,
        real_token_reserves=783_100_000 * 10**6,
        complete=False,
    )
    bare = PaperExecutor(slippage_bps=100.0)
    realistic = PaperExecutor(slippage_bps=100.0, realism=_realism())
    bare_fill = bare.sell("M", 5_000, funded_curve)
    real_fill = realistic.sell("M", 5_000, funded_curve)
    assert real_fill.notional_sol < bare_fill.notional_sol
    assert real_fill.pump_fee_sol > 0.0


def test_realism_buy_fail_prob_can_force_failure() -> None:
    exe = PaperExecutor(slippage_bps=0.0, realism=_realism(buy_fail_prob=1.0))
    with pytest.raises(PaperFillFailedError, match="simulated tx fail"):
        exe.buy("M", 1000, _fresh_curve())


def test_realism_sell_fail_prob_can_force_failure() -> None:
    funded_curve = BondingCurveState(
        virtual_sol_reserves=DEFAULT_VIRTUAL_SOL_RESERVES + 5 * LAMPORTS_PER_SOL,
        virtual_token_reserves=DEFAULT_VIRTUAL_TOKEN_RESERVES - 5_000_000_000,
        real_sol_reserves=5 * LAMPORTS_PER_SOL,
        real_token_reserves=788_100_000 * 10**6,
        complete=False,
    )
    exe = PaperExecutor(slippage_bps=0.0, realism=_realism(sell_fail_prob=1.0))
    with pytest.raises(PaperFillFailedError):
        exe.sell("M", 1000, funded_curve)


def test_realism_fail_prob_zero_never_fails() -> None:
    exe = PaperExecutor(
        slippage_bps=0.0,
        realism=_realism(buy_fail_prob=0.0, sell_fail_prob=0.0),
    )
    # 50 attempts in a row should never raise.
    for _ in range(50):
        exe.buy("M", 1_000, _fresh_curve())


def test_realism_latency_drift_makes_buys_pay_more_when_inflow_positive() -> None:
    exe = PaperExecutor(slippage_bps=0.0, realism=_realism())
    curve = _fresh_curve()
    quiet = exe.buy("M", 50_000, curve, observed_inflow_sol_per_sec=0.0)
    # Reset RNG so the latency / priority-fee draws are identical between
    # the two calls — only the inflow argument differs.
    exe._rng = random.Random(1)
    busy = exe.buy("M", 50_000, curve, observed_inflow_sol_per_sec=2.0)
    assert busy.drift_sol_absorbed > quiet.drift_sol_absorbed
    assert busy.fill_price_sol_per_token > quiet.fill_price_sol_per_token


def test_realism_seeded_rng_is_deterministic() -> None:
    """Same seed → same fill, so backtests are reproducible."""
    cfg = _realism(rng_seed=42, buy_fail_prob=0.0)
    a = PaperExecutor(slippage_bps=75.0, realism=cfg)
    b = PaperExecutor(slippage_bps=75.0, realism=cfg)
    curve = _fresh_curve()
    fa = a.buy("M", 10_000, curve, observed_inflow_sol_per_sec=1.0)
    fb = b.buy("M", 10_000, curve, observed_inflow_sol_per_sec=1.0)
    assert fa.notional_sol == pytest.approx(fb.notional_sol)
    assert fa.latency_ms == pytest.approx(fb.latency_ms)
    assert fa.priority_fee_sol == pytest.approx(fb.priority_fee_sol)


def test_lognormal_recovers_median_in_expectation() -> None:
    rng = random.Random(0)
    samples = [_sample_lognormal(rng, p50=1000.0, p99=10_000.0) for _ in range(20_000)]
    samples.sort()
    p50_est = samples[len(samples) // 2]
    # Empirical median should be within ~5% of the configured 1000.
    assert abs(p50_est - 1000.0) / 1000.0 < 0.05


def test_advance_curve_by_sol_increases_price() -> None:
    curve = _fresh_curve()
    advanced = _advance_curve_by_sol(curve, 5.0)
    assert advanced.real_sol_reserves > curve.real_sol_reserves
    assert advanced.virtual_token_reserves < curve.virtual_token_reserves
    assert advanced.price_per_token_sol() > curve.price_per_token_sol()


def test_advance_curve_by_zero_is_noop() -> None:
    curve = _fresh_curve()
    advanced = _advance_curve_by_sol(curve, 0.0)
    assert advanced is curve
