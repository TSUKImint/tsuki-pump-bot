"""Paper executor — simulated buys/sells against the real on-chain bonding curve.

We never submit a transaction. We read the bonding-curve state from RPC (or
synthesize it from observed buy events in paper-mock mode), compute the
expected fill price using the constant-product math from `bonding_curve`,
apply a configurable slippage, and record a fill in the event store.

The fill structure is the same shape a live executor would emit so the rest
of the pipeline (risk engine, position monitor, dashboard) is unchanged
when we eventually plug in a live executor.

When `PaperRealismConfig.enabled` is true the executor additionally models
the things that make paper P&L diverge from live P&L on pump.fun:

* **Latency-driven curve drift** — between detection and "landing" we
  sample an end-to-end latency from a log-normal interpolated between the
  configured p50 and p99 in ms. During that latency, other buyers push the
  curve forward at the caller-provided ``observed_inflow_sol_per_sec``
  rate, so our fill price is materially worse than the spot we saw.
* **Pump.fun protocol fee** — 1% of notional (configurable) on every buy
  and sell, charged on top of slippage.
* **Priority fee** — sampled per fill (log-normal between p50 and p99)
  and added to buy cost / deducted from sell proceeds. Models the cost of
  "winning the slot" on a congested chain.
* **Transaction failure** — Bernoulli draw at ``buy_fail_prob`` /
  ``sell_fail_prob``. Failed attempts raise `PaperFillFailedError` so the
  orchestrator records them as `paper.buy_failed` / `paper.sell_failed`
  events (it already handles those event kinds).

When realism is disabled the executor behaves exactly as before — the
original tests continue to pass without modification.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import structlog

from ..config import PaperRealismConfig
from ..solana.bonding_curve import (
    LAMPORTS_PER_SOL,
    TOKEN_UNIT_MULTIPLIER,
    BondingCurveState,
    buy_cost_sol,
    sell_proceeds_sol,
)

logger = structlog.get_logger(__name__)


class PaperFillFailedError(ValueError):
    """Raised when a paper fill is "rejected" by the simulated chain.

    Subclasses ValueError so the existing orchestrator handler
    (`except ValueError`) keeps working without modification.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class PaperFill:
    mint: str
    side: str  # BUY | SELL
    requested_units: float
    fill_units: float
    fill_price_sol_per_token: float
    notional_sol: float
    slippage_bps_applied: float
    # Realism diagnostics (zero when realism is disabled).
    pump_fee_sol: float = 0.0
    priority_fee_sol: float = 0.0
    latency_ms: float = 0.0
    drift_sol_absorbed: float = 0.0
    paper: bool = True


class PaperExecutor:
    """Deterministic paper fills (with optional pump.fun realism layer)."""

    def __init__(
        self,
        slippage_bps: float,
        realism: PaperRealismConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        self.slippage_bps = slippage_bps
        self.realism = realism or PaperRealismConfig()
        if rng is not None:
            self._rng = rng
        elif self.realism.rng_seed is not None:
            self._rng = random.Random(self.realism.rng_seed)
        else:
            self._rng = random.Random()

    # ── public API ────────────────────────────────────────────────────────

    def buy(
        self,
        mint: str,
        units: float,
        curve: BondingCurveState,
        *,
        observed_inflow_sol_per_sec: float = 0.0,
    ) -> PaperFill:
        if units <= 0:
            raise ValueError("units must be positive")

        # 1) Failure draw (only when realism is on).
        if self.realism.enabled and self._rng.random() < self.realism.buy_fail_prob:
            raise PaperFillFailedError("simulated tx fail (slippage / blockhash)")

        # 2) Advance the curve by the SOL other buyers absorbed during our
        #    end-to-end latency window.
        latency_ms = self._sample_latency_ms()
        drift_sol = max(0.0, observed_inflow_sol_per_sec) * (latency_ms / 1000.0)
        drifted_curve = _advance_curve_by_sol(curve, drift_sol) if self.realism.enabled else curve

        # 3) Bonding-curve cost on the drifted curve.
        raw_cost = buy_cost_sol(drifted_curve, units)
        if raw_cost == float("inf") or raw_cost <= 0:
            raise ValueError("curve cannot fulfil this order (insufficient depth)")

        # 4) Slippage and pump.fun fee.
        slippage_adj = raw_cost * (1.0 + self.slippage_bps / 10_000.0)
        pump_fee_sol = (
            slippage_adj * (self.realism.pump_fee_bps / 10_000.0) if self.realism.enabled else 0.0
        )
        # 5) Priority fee (sampled per fill).
        priority_fee_sol = self._sample_priority_fee_sol() if self.realism.enabled else 0.0

        notional = slippage_adj + pump_fee_sol + priority_fee_sol
        fill_price = notional / units
        return PaperFill(
            mint=mint,
            side="BUY",
            requested_units=units,
            fill_units=units,
            fill_price_sol_per_token=fill_price,
            notional_sol=notional,
            slippage_bps_applied=self.slippage_bps,
            pump_fee_sol=pump_fee_sol,
            priority_fee_sol=priority_fee_sol,
            latency_ms=latency_ms if self.realism.enabled else 0.0,
            drift_sol_absorbed=drift_sol if self.realism.enabled else 0.0,
        )

    def sell(
        self,
        mint: str,
        units: float,
        curve: BondingCurveState,
        *,
        observed_inflow_sol_per_sec: float = 0.0,
    ) -> PaperFill:
        if units <= 0:
            raise ValueError("units must be positive")

        # 1) Failure draw.
        if self.realism.enabled and self._rng.random() < self.realism.sell_fail_prob:
            raise PaperFillFailedError("simulated tx fail (slippage / blockhash)")

        # 2) Curve drift — for a sell, other buyers' inflow actually *helps*
        #    us (price rises). We still advance the curve forward but the
        #    sign of the effect flips through the math naturally.
        latency_ms = self._sample_latency_ms()
        drift_sol = max(0.0, observed_inflow_sol_per_sec) * (latency_ms / 1000.0)
        drifted_curve = _advance_curve_by_sol(curve, drift_sol) if self.realism.enabled else curve

        raw_proceeds = sell_proceeds_sol(drifted_curve, units)
        # Slippage on a sell *reduces* proceeds.
        slippage_adj = raw_proceeds * (1.0 - self.slippage_bps / 10_000.0)
        pump_fee_sol = (
            slippage_adj * (self.realism.pump_fee_bps / 10_000.0) if self.realism.enabled else 0.0
        )
        priority_fee_sol = self._sample_priority_fee_sol() if self.realism.enabled else 0.0

        # Sell-side: protocol fee + priority fee are *deducted* from proceeds.
        notional = max(0.0, slippage_adj - pump_fee_sol - priority_fee_sol)
        fill_price = (notional / units) if units else 0.0
        return PaperFill(
            mint=mint,
            side="SELL",
            requested_units=units,
            fill_units=units,
            fill_price_sol_per_token=fill_price,
            notional_sol=notional,
            slippage_bps_applied=self.slippage_bps,
            pump_fee_sol=pump_fee_sol,
            priority_fee_sol=priority_fee_sol,
            latency_ms=latency_ms if self.realism.enabled else 0.0,
            drift_sol_absorbed=drift_sol if self.realism.enabled else 0.0,
        )

    # ── helpers ───────────────────────────────────────────────────────────

    def _sample_latency_ms(self) -> float:
        """Sample end-to-end latency from a log-normal calibrated to (p50, p99).

        We fit mu = ln(p50) and sigma so that the 99th percentile of the
        log-normal matches the configured p99. This gives a heavy right tail
        — exactly what users observe on Solana when a slot is congested.
        """
        if not self.realism.enabled:
            return 0.0
        p50 = max(1.0, float(self.realism.end_to_end_latency_ms_p50))
        p99 = max(p50, float(self.realism.end_to_end_latency_ms_p99))
        return _sample_lognormal(self._rng, p50, p99)

    def _sample_priority_fee_sol(self) -> float:
        """Sample priority fee in SOL from a log-normal calibrated to (p50, p99)."""
        p50 = max(1.0, float(self.realism.priority_fee_lamports_p50))
        p99 = max(p50, float(self.realism.priority_fee_lamports_p99))
        lamports = _sample_lognormal(self._rng, p50, p99)
        return lamports / LAMPORTS_PER_SOL


# ── module-level helpers ───────────────────────────────────────────────────


def _sample_lognormal(rng: random.Random, p50: float, p99: float) -> float:
    """Draw from a log-normal with the given median and 99th percentile.

    Closed-form: ln(p50) = mu, ln(p99) = mu + 2.3263*sigma → sigma derived.
    """
    if p50 <= 0:
        return 0.0
    if p99 <= p50:
        return p50
    mu = math.log(p50)
    sigma = (math.log(p99) - mu) / 2.3263478740408408  # qnorm(0.99)
    return float(math.exp(rng.gauss(mu, sigma)))


def _advance_curve_by_sol(curve: BondingCurveState, sol_to_absorb: float) -> BondingCurveState:
    """Return a new curve advanced by `sol_to_absorb` of buyer inflow.

    Models the impact of other traders' buys during our latency window. Uses
    the same constant-product invariant as `buy_cost_sol` in reverse: given
    we know the SOL going in, compute the new virtual reserves.
    """
    if sol_to_absorb <= 0:
        return curve
    lamports_in = int(sol_to_absorb * LAMPORTS_PER_SOL)
    new_virtual_sol = curve.virtual_sol_reserves + lamports_in
    k = curve.virtual_sol_reserves * curve.virtual_token_reserves
    # Constant product → new virtual tokens shrinks.
    new_virtual_tokens = int(k / max(1, new_virtual_sol))
    # Real reserves shift correspondingly. Real SOL grows by `lamports_in`;
    # real tokens shrink by the same fraction as the virtual side.
    delta_tokens = curve.virtual_token_reserves - new_virtual_tokens
    new_real_tokens = max(0, curve.real_token_reserves - delta_tokens)
    new_real_sol = curve.real_sol_reserves + lamports_in
    # Sanity clamp — token side cannot go negative.
    if new_virtual_tokens <= TOKEN_UNIT_MULTIPLIER:
        return curve  # would empty the curve; treat as no-op
    return BondingCurveState(
        virtual_sol_reserves=new_virtual_sol,
        virtual_token_reserves=new_virtual_tokens,
        real_sol_reserves=new_real_sol,
        real_token_reserves=new_real_tokens,
        complete=curve.complete,
    )
