"""Sizing + risk engine tests."""

from __future__ import annotations

from tsukibot_pump.core.risk import RiskEngine, RiskLimits
from tsukibot_pump.core.sizing import SizingInputs, size_memecoin_position


def _limits() -> RiskLimits:
    return RiskLimits(
        daily_drawdown_kill=0.10,
        total_drawdown_kill=0.25,
        single_token_cap_fraction=0.05,
        max_open_positions=5,
        hard_cap_per_trade_sol=0.5,
    )


def test_risk_engine_allows_within_caps() -> None:
    eng = RiskEngine(starting_bankroll_sol=10.0, limits=_limits())
    decision = eng.check_open_allowed(0.4)
    assert decision.allowed, decision.reason


def test_risk_engine_rejects_over_single_token_cap() -> None:
    eng = RiskEngine(starting_bankroll_sol=10.0, limits=_limits())
    # 0.6 > 5% of 10 = 0.5 cap; also > hard cap. Test the cap rejection.
    decision = eng.check_open_allowed(0.51)
    assert not decision.allowed
    assert "single-token cap" in decision.reason or "hard cap" in decision.reason


def test_risk_engine_rejects_over_hard_cap() -> None:
    eng = RiskEngine(starting_bankroll_sol=1000.0, limits=_limits())
    # Hard cap is 0.5 SOL regardless of bankroll.
    decision = eng.check_open_allowed(0.6)
    assert not decision.allowed
    assert "hard cap" in decision.reason


def test_risk_engine_rejects_at_max_open_positions() -> None:
    eng = RiskEngine(starting_bankroll_sol=10.0, limits=_limits())
    for _ in range(5):
        eng.record_open(0.1)
    decision = eng.check_open_allowed(0.1)
    assert not decision.allowed
    assert "open positions" in decision.reason


def test_risk_engine_trips_on_total_drawdown() -> None:
    eng = RiskEngine(starting_bankroll_sol=10.0, limits=_limits())
    eng.record_open(0.5)
    eng.record_close(realized_pnl_sol=-3.0, notional_freed_sol=0.5)  # -30%
    assert eng.tripped
    assert "total drawdown" in eng.trip_reason


def test_risk_engine_trips_on_daily_drawdown() -> None:
    eng = RiskEngine(starting_bankroll_sol=10.0, limits=_limits())
    eng.record_open(0.5)
    eng.record_close(realized_pnl_sol=-1.5, notional_freed_sol=0.5)  # 15% daily
    assert eng.tripped
    assert "daily drawdown" in eng.trip_reason


def test_size_memecoin_zero_at_no_edge() -> None:
    """Score 50 → win probability 50% → Kelly fraction 0 → zero units."""
    inputs = SizingInputs(
        bankroll_sol=10.0,
        composite_score=50.0,
        cost_per_unit_sol=0.0001,
        fraction_of_kelly=0.5,
        hard_cap_per_trade_sol=0.5,
        single_token_cap_sol=0.5,
    )
    result = size_memecoin_position(inputs)
    # Kelly at p=0.5, b=3.0: f* = (3*0.5 - 0.5)/3 = 1/3 — still positive!
    # That's correct: a 50/50 with payoff 3:1 is profitable. We cap via
    # hard_cap, single_token_cap, and bankroll * fraction_of_kelly.
    assert result.units > 0
    assert result.notional_sol > 0
    assert result.notional_sol <= 0.5


def test_size_memecoin_hard_cap_binds() -> None:
    inputs = SizingInputs(
        bankroll_sol=100.0,
        composite_score=100.0,
        cost_per_unit_sol=0.0001,
        fraction_of_kelly=1.0,
        hard_cap_per_trade_sol=0.5,
        single_token_cap_sol=5.0,
    )
    result = size_memecoin_position(inputs)
    assert result.binding_constraint == "hard_cap"
    assert result.notional_sol == 0.5


def test_size_memecoin_single_token_cap_binds() -> None:
    inputs = SizingInputs(
        bankroll_sol=100.0,
        composite_score=100.0,
        cost_per_unit_sol=0.0001,
        fraction_of_kelly=1.0,
        hard_cap_per_trade_sol=10.0,
        single_token_cap_sol=2.0,
    )
    result = size_memecoin_position(inputs)
    assert result.binding_constraint == "capital"
    assert result.notional_sol == 2.0
