"""Pump.fun bonding curve math.

The pump.fun bonding curve is a fixed-product (x*y=k) virtual reserve curve
deployed at token creation. Each token starts with virtual reserves that
make the initial price tiny and migrates to PumpSwap (since March 2025;
previously Raydium) when the SOL side of the curve crosses ~85 SOL.

Reference values (verified from multiple analyses including ChainCatcher,
Bullrank, and Quicknode's pump.fun bot guide, 2025-2026):

  Virtual reserve at deploy:
    virtual_sol_reserves      = 30 SOL                     (30e9 lamports)
    virtual_token_reserves    = 1_073_000_000 tokens       (1.073e15 with 6 dec)
    real_sol_reserves_at_dep  = 0
    real_token_reserves_at_dep = 793_100_000 tokens

  Graduation triggers when real_sol_reserves crosses ~85 SOL, equivalent to
  a market cap of ~$69k (varies with SOL price), at which point the LP is
  migrated and trading switches to the PumpSwap AMM.

Constants below are scoped as defaults — the actual on-chain state for a
specific token can be read from its BondingCurve account and overrides
these when computing real prices. We expose the *math* here so curve
prediction (filter) can simulate forward without an RPC roundtrip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

LAMPORTS_PER_SOL: Final[int] = 1_000_000_000
TOKEN_DECIMALS: Final[int] = 6
TOKEN_UNIT_MULTIPLIER: Final[int] = 10**TOKEN_DECIMALS

# Default virtual reserves at deploy (overridden by per-token reads).
DEFAULT_VIRTUAL_SOL_RESERVES: Final[int] = 30 * LAMPORTS_PER_SOL  # 30 SOL in lamports
DEFAULT_VIRTUAL_TOKEN_RESERVES: Final[int] = 1_073_000_000 * TOKEN_UNIT_MULTIPLIER

# Approximate graduation threshold. Real graduation is signalled by the
# `complete` flag on the BondingCurve account; this value is for curve-
# prediction math only.
DEFAULT_GRADUATION_SOL_THRESHOLD: Final[float] = 85.0


@dataclass(frozen=True, slots=True)
class BondingCurveState:
    """Snapshot of a pump.fun BondingCurve account."""

    virtual_sol_reserves: int  # lamports
    virtual_token_reserves: int  # token base units (with TOKEN_DECIMALS)
    real_sol_reserves: int  # lamports
    real_token_reserves: int  # token base units
    complete: bool  # true after graduation; trading moves to PumpSwap

    @property
    def real_sol_in_curve(self) -> float:
        return self.real_sol_reserves / LAMPORTS_PER_SOL

    def price_per_token_sol(self) -> float:
        """Spot price (SOL per token) from virtual reserves.

        Uses the virtual reserves (which include the initial offsets) so the
        first-buy price is well-defined.
        """
        if self.virtual_token_reserves <= 0:
            return 0.0
        return (self.virtual_sol_reserves / LAMPORTS_PER_SOL) / (
            self.virtual_token_reserves / TOKEN_UNIT_MULTIPLIER
        )


def buy_cost_sol(
    state: BondingCurveState,
    token_units_to_buy: float,
) -> float:
    """Return SOL cost (excluding fees) to buy `token_units_to_buy` tokens.

    Uses constant-product invariant: k = virtual_sol * virtual_tokens.
    cost = new_sol - old_sol, where new_sol = k / (virtual_tokens - units).

    Returns float SOL (not lamports).
    """
    if token_units_to_buy <= 0:
        return 0.0
    base_units = token_units_to_buy * TOKEN_UNIT_MULTIPLIER
    if base_units >= state.virtual_token_reserves:
        # Asking for more than the curve has → infinite cost (graduate first).
        return float("inf")
    k = state.virtual_sol_reserves * state.virtual_token_reserves
    new_virtual_sol = k / (state.virtual_token_reserves - base_units)
    cost_lamports = new_virtual_sol - state.virtual_sol_reserves
    return cost_lamports / LAMPORTS_PER_SOL


def sell_proceeds_sol(
    state: BondingCurveState,
    token_units_to_sell: float,
) -> float:
    """Return SOL proceeds (excluding fees) from selling `token_units_to_sell`.

    Constant-product in reverse. Returns float SOL.
    """
    if token_units_to_sell <= 0:
        return 0.0
    base_units = token_units_to_sell * TOKEN_UNIT_MULTIPLIER
    k = state.virtual_sol_reserves * state.virtual_token_reserves
    new_virtual_sol = k / (state.virtual_token_reserves + base_units)
    proceeds_lamports = state.virtual_sol_reserves - new_virtual_sol
    # Proceeds cannot exceed real SOL reserves (you can't withdraw virtual).
    proceeds_lamports = min(proceeds_lamports, state.real_sol_reserves)
    return max(0.0, proceeds_lamports / LAMPORTS_PER_SOL)


def sol_needed_to_reach_graduation(
    state: BondingCurveState,
    target_real_sol: float = DEFAULT_GRADUATION_SOL_THRESHOLD,
) -> float:
    """Estimate additional SOL needed (across all buyers) to graduate.

    Returns a non-negative SOL amount. 0 if already past threshold.
    """
    remaining = target_real_sol - state.real_sol_in_curve
    return max(0.0, remaining)


def curve_progress_fraction(
    state: BondingCurveState,
    target_real_sol: float = DEFAULT_GRADUATION_SOL_THRESHOLD,
) -> float:
    """Curve progress as a fraction in [0, 1]."""
    if target_real_sol <= 0:
        return 1.0
    return max(0.0, min(1.0, state.real_sol_in_curve / target_real_sol))
