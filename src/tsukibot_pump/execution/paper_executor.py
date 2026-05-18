"""Paper executor — simulated buys/sells against the real on-chain bonding curve.

We never submit a transaction. We read the bonding-curve state from RPC (or
synthesize it from observed buy events in paper-mock mode), compute the
expected fill price using the constant-product math from `bonding_curve`,
apply a configurable slippage, and record a fill in the event store.

The fill structure is the same shape a live executor would emit so the rest
of the pipeline (risk engine, position monitor, dashboard) is unchanged
when we eventually plug in a live executor.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from ..solana.bonding_curve import BondingCurveState, buy_cost_sol, sell_proceeds_sol

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PaperFill:
    mint: str
    side: str  # BUY | SELL
    requested_units: float
    fill_units: float
    fill_price_sol_per_token: float
    notional_sol: float
    slippage_bps_applied: float
    paper: bool = True


class PaperExecutor:
    """Deterministic paper fills."""

    def __init__(self, slippage_bps: float) -> None:
        if slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        self.slippage_bps = slippage_bps

    def buy(
        self,
        mint: str,
        units: float,
        curve: BondingCurveState,
    ) -> PaperFill:
        if units <= 0:
            raise ValueError("units must be positive")
        raw_cost = buy_cost_sol(curve, units)
        if raw_cost == float("inf") or raw_cost <= 0:
            raise ValueError("curve cannot fulfil this order (insufficient depth)")
        adjusted = raw_cost * (1.0 + self.slippage_bps / 10_000.0)
        fill_price = adjusted / units
        return PaperFill(
            mint=mint,
            side="BUY",
            requested_units=units,
            fill_units=units,
            fill_price_sol_per_token=fill_price,
            notional_sol=adjusted,
            slippage_bps_applied=self.slippage_bps,
        )

    def sell(
        self,
        mint: str,
        units: float,
        curve: BondingCurveState,
    ) -> PaperFill:
        if units <= 0:
            raise ValueError("units must be positive")
        raw_proceeds = sell_proceeds_sol(curve, units)
        # Slippage on a sell *reduces* proceeds.
        adjusted = raw_proceeds * (1.0 - self.slippage_bps / 10_000.0)
        fill_price = (adjusted / units) if units else 0.0
        return PaperFill(
            mint=mint,
            side="SELL",
            requested_units=units,
            fill_units=units,
            fill_price_sol_per_token=fill_price,
            notional_sol=adjusted,
            slippage_bps_applied=self.slippage_bps,
        )
