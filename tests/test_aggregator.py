"""TokenStateAggregator tests."""

from __future__ import annotations

from tsukibot_pump.scout.aggregator import TokenStateAggregator
from tsukibot_pump.solana.pump_program import PumpEvent, PumpInstructionKind


def _event(
    kind: PumpInstructionKind,
    mint: str,
    *,
    actor: str | None = None,
    dev: str | None = None,
    slot: int = 1,
    ts: int = 1_700_000_000,
    sol: float | None = None,
    sig: str = "sig",
) -> PumpEvent:
    return PumpEvent(
        kind=kind,
        signature=sig,
        slot=slot,
        block_time_unix=ts,
        mint=mint,
        dev_wallet=dev,
        actor_wallet=actor,
        sol_amount=sol,
        token_amount=None,
    )


def test_create_then_buy_populates_state() -> None:
    agg = TokenStateAggregator()
    agg.ingest(_event(PumpInstructionKind.CREATE, "M1", dev="DEV1", ts=100))
    agg.ingest(_event(PumpInstructionKind.BUY, "M1", actor="W1", slot=2, ts=110, sol=0.5))
    state = agg.get("M1")
    assert state is not None
    assert state.dev_wallet == "DEV1"
    assert state.created_at_unix == 100
    assert state.sol_in_curve == 0.5
    assert len(state.buys) == 1
    assert state.buys[0].wallet == "W1"


def test_velocity_sol_per_min_over_window() -> None:
    agg = TokenStateAggregator()
    agg.ingest(_event(PumpInstructionKind.CREATE, "M2", dev="DEV", ts=1000))
    agg.ingest(_event(PumpInstructionKind.BUY, "M2", actor="A", ts=1000, sol=1.0))
    agg.ingest(_event(PumpInstructionKind.BUY, "M2", actor="B", ts=1030, sol=1.0))
    agg.ingest(_event(PumpInstructionKind.BUY, "M2", actor="C", ts=1060, sol=1.0))
    state = agg.get("M2")
    assert state is not None
    # 3 SOL across 60s → 3 SOL/min.
    assert 2.9 < state.last_sol_velocity_sol_per_min < 3.1


def test_distinct_buyers_60s_window() -> None:
    agg = TokenStateAggregator()
    agg.ingest(_event(PumpInstructionKind.CREATE, "M3", dev="DEV"))
    for i, ts in enumerate([1000, 1010, 1020, 1100]):  # last one outside 60s of t=1100
        agg.ingest(_event(PumpInstructionKind.BUY, "M3", actor=f"W{i}", ts=ts, sol=0.1))
    state = agg.get("M3")
    assert state is not None
    # At the last event (ts=1100), 60s window cuts at 1040.
    # Buyers seen: W0(1000) excluded, W1(1010) excluded, W2(1020) excluded, W3(1100) included.
    assert state.distinct_buyers_60s == 1


def test_lru_eviction() -> None:
    agg = TokenStateAggregator(max_tokens=3)
    for i in range(5):
        agg.ingest(_event(PumpInstructionKind.CREATE, f"M{i}", dev=f"D{i}", ts=i))
    assert len(agg) == 3
    # Oldest two (M0, M1) should have been evicted.
    assert agg.get("M0") is None
    assert agg.get("M1") is None
    assert agg.get("M4") is not None


def test_dev_sell_updates_cumulative_pulled() -> None:
    agg = TokenStateAggregator()
    agg.ingest(_event(PumpInstructionKind.CREATE, "M4", dev="DEV"))
    agg.ingest(_event(PumpInstructionKind.BUY, "M4", actor="X", ts=10, sol=2.0))
    agg.ingest(_event(PumpInstructionKind.SELL, "M4", actor="DEV", dev="DEV", ts=20, sol=1.5))
    state = agg.get("M4")
    assert state is not None
    assert state.dev_activity.cumulative_sol_pulled == 1.5
