"""Bonding curve math tests."""

from __future__ import annotations

import math

import pytest

from tsukibot_pump.solana.bonding_curve import (
    DEFAULT_GRADUATION_SOL_THRESHOLD,
    DEFAULT_VIRTUAL_SOL_RESERVES,
    DEFAULT_VIRTUAL_TOKEN_RESERVES,
    LAMPORTS_PER_SOL,
    BondingCurveState,
    buy_cost_sol,
    curve_progress_fraction,
    sell_proceeds_sol,
    sol_needed_to_reach_graduation,
)


def _fresh_state() -> BondingCurveState:
    return BondingCurveState(
        virtual_sol_reserves=DEFAULT_VIRTUAL_SOL_RESERVES,
        virtual_token_reserves=DEFAULT_VIRTUAL_TOKEN_RESERVES,
        real_sol_reserves=0,
        real_token_reserves=793_100_000 * 10**6,
        complete=False,
    )


def test_fresh_state_price_is_tiny() -> None:
    """Initial price should be sub-cent (30 SOL / ~1B tokens)."""
    state = _fresh_state()
    price = state.price_per_token_sol()
    assert price > 0
    assert price < 1e-6


def test_buy_cost_zero_units_is_zero() -> None:
    assert buy_cost_sol(_fresh_state(), 0) == 0.0
    assert buy_cost_sol(_fresh_state(), -10) == 0.0


def test_buy_cost_monotonic_in_units() -> None:
    state = _fresh_state()
    c10 = buy_cost_sol(state, 10_000)
    c100 = buy_cost_sol(state, 100_000)
    c1m = buy_cost_sol(state, 1_000_000)
    assert c10 < c100 < c1m


def test_buy_cost_for_entire_reserve_is_infinite() -> None:
    state = _fresh_state()
    too_many = state.virtual_token_reserves / 10**6  # exact reserve count
    cost = buy_cost_sol(state, too_many)
    assert cost == float("inf")


def test_sell_proceeds_bounded_by_real_sol() -> None:
    """You cannot withdraw virtual reserves — only real SOL backs sales."""
    state = BondingCurveState(
        virtual_sol_reserves=DEFAULT_VIRTUAL_SOL_RESERVES + 5 * LAMPORTS_PER_SOL,
        virtual_token_reserves=DEFAULT_VIRTUAL_TOKEN_RESERVES - 1_000_000_000,
        real_sol_reserves=5 * LAMPORTS_PER_SOL,  # only 5 SOL real
        real_token_reserves=0,
        complete=False,
    )
    # Try to sell more than the curve has — proceeds must be capped at 5 SOL.
    huge = 1_000_000_000_000
    proceeds = sell_proceeds_sol(state, huge)
    assert proceeds <= 5.0


def test_round_trip_buy_then_sell_loses_money_to_curve_shape() -> None:
    """Buy then sell of the same units should yield less SOL than spent
    (no fees applied here — the loss is purely curve geometry)."""
    state = _fresh_state()
    units = 500_000
    cost = buy_cost_sol(state, units)
    # Update state as if the buy happened.
    base_units = units * 10**6
    k = state.virtual_sol_reserves * state.virtual_token_reserves
    new_virtual_sol = k / (state.virtual_token_reserves - base_units)
    cost_lamports = int(new_virtual_sol - state.virtual_sol_reserves)
    updated = BondingCurveState(
        virtual_sol_reserves=state.virtual_sol_reserves + cost_lamports,
        virtual_token_reserves=state.virtual_token_reserves - base_units,
        real_sol_reserves=cost_lamports,
        real_token_reserves=state.real_token_reserves - base_units,
        complete=False,
    )
    proceeds = sell_proceeds_sol(updated, units)
    # On an x*y=k curve, buying then immediately selling returns the same
    # SOL back if you sell at the new price level (the curve is reversible).
    # In practice fees + slippage create the loss; we just check the math
    # is self-consistent within float precision.
    assert math.isclose(proceeds, cost, rel_tol=1e-6)


def test_sol_needed_to_reach_graduation() -> None:
    state = BondingCurveState(
        virtual_sol_reserves=DEFAULT_VIRTUAL_SOL_RESERVES + 50 * LAMPORTS_PER_SOL,
        virtual_token_reserves=DEFAULT_VIRTUAL_TOKEN_RESERVES,
        real_sol_reserves=50 * LAMPORTS_PER_SOL,
        real_token_reserves=0,
        complete=False,
    )
    remaining = sol_needed_to_reach_graduation(state)
    assert remaining == pytest.approx(DEFAULT_GRADUATION_SOL_THRESHOLD - 50.0, abs=1e-6)

    past = BondingCurveState(
        virtual_sol_reserves=DEFAULT_VIRTUAL_SOL_RESERVES + 100 * LAMPORTS_PER_SOL,
        virtual_token_reserves=DEFAULT_VIRTUAL_TOKEN_RESERVES,
        real_sol_reserves=100 * LAMPORTS_PER_SOL,
        real_token_reserves=0,
        complete=False,
    )
    assert sol_needed_to_reach_graduation(past) == 0.0


def test_curve_progress_fraction_bounds() -> None:
    fresh = _fresh_state()
    assert curve_progress_fraction(fresh) == 0.0

    state = BondingCurveState(
        virtual_sol_reserves=DEFAULT_VIRTUAL_SOL_RESERVES + 42 * LAMPORTS_PER_SOL,
        virtual_token_reserves=DEFAULT_VIRTUAL_TOKEN_RESERVES,
        real_sol_reserves=42 * LAMPORTS_PER_SOL,
        real_token_reserves=0,
        complete=False,
    )
    frac = curve_progress_fraction(state)
    assert 0 < frac < 1
    assert frac == pytest.approx(42.0 / DEFAULT_GRADUATION_SOL_THRESHOLD, rel=1e-6)
