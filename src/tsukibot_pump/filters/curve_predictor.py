"""Filter 5 — bonding curve graduation predictor.

The pump.fun bonding curve is deterministic math (constant-product virtual
reserves; see `solana.bonding_curve`). The graduation event at ~85 SOL
triggers an LP migration to PumpSwap (since March 2025) which historically
correlates with a price pop because (a) new buyers see the token on
PumpSwap dashboards, (b) the LP move itself adds price-supportive activity.

We don't need raw speed for this — graduation happens over minutes, not
milliseconds. The edge is: enter just before graduation when velocity is
strong, exit shortly after.

Score semantics:
  - Pre-curve-threshold: score = 40 (no edge yet).
  - Past threshold AND velocity above min AND distinct-buyer floor met:
    score scales from 60 (just past) to 100 (within 10% of graduation).
  - Already graduated: score = 50 (curve filter doesn't apply post-graduation).
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from ..config import CurveGraduationConfig
from ..models import FilterOutcome, TokenState
from ..solana.bonding_curve import DEFAULT_GRADUATION_SOL_THRESHOLD

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CurveSnapshot:
    sol_in_curve: float
    velocity_sol_per_min: float
    distinct_buyers_60s: int
    complete: bool


class CurvePredictor:
    """Score based on proximity to graduation + curve velocity."""

    NAME = "curve_graduation"

    def __init__(self, config: CurveGraduationConfig) -> None:
        self.config = config

    def evaluate(self, token: TokenState) -> FilterOutcome:
        if not self.config.enabled:
            return FilterOutcome(name=self.NAME, score=50.0, notes="disabled")

        snap = CurveSnapshot(
            sol_in_curve=token.sol_in_curve,
            velocity_sol_per_min=token.last_sol_velocity_sol_per_min,
            distinct_buyers_60s=token.distinct_buyers_60s,
            complete=token.curve_complete,
        )

        if snap.complete:
            return FilterOutcome(name=self.NAME, score=50.0, notes="already graduated")

        if snap.sol_in_curve < self.config.enter_after_sol_in_curve_gte:
            return FilterOutcome(
                name=self.NAME,
                score=40.0,
                notes=f"curve at {snap.sol_in_curve:.1f} SOL (< threshold)",
            )

        if snap.velocity_sol_per_min < self.config.min_velocity_sol_per_min:
            return FilterOutcome(
                name=self.NAME,
                score=45.0,
                notes=(
                    f"velocity {snap.velocity_sol_per_min:.2f} SOL/min "
                    f"(< {self.config.min_velocity_sol_per_min})"
                ),
            )

        if snap.distinct_buyers_60s < self.config.min_distinct_buyers_60s:
            return FilterOutcome(
                name=self.NAME,
                score=40.0,
                notes=(
                    f"only {snap.distinct_buyers_60s} distinct buyers in 60s "
                    f"(< {self.config.min_distinct_buyers_60s})"
                ),
            )

        # All gates passed: scale 60 → 100 as we approach graduation.
        target = DEFAULT_GRADUATION_SOL_THRESHOLD
        progress = (snap.sol_in_curve - self.config.enter_after_sol_in_curve_gte) / max(
            1e-6, target - self.config.enter_after_sol_in_curve_gte
        )
        progress = max(0.0, min(1.0, progress))
        score = 60.0 + 40.0 * progress
        return FilterOutcome(
            name=self.NAME,
            score=score,
            notes=(
                f"curve {snap.sol_in_curve:.1f}/{target:.0f} SOL, "
                f"velocity {snap.velocity_sol_per_min:.2f} SOL/min, "
                f"{snap.distinct_buyers_60s} buyers/60s"
            ),
        )
