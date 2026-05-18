"""Composite scorer.

Two interchangeable scorers, selected by `ScoringConfig.mode`:

  - `CompositeScorer`             (default; mode="weighted"). v0.2 behaviour:
    weighted sum of FilterOutcome scores using ScoringWeights. Hard-rejected
    if any filter hard-rejects.

  - `GraduationProbabilityScorer` (v0.3; mode="graduation_probability"). A
    logistic transform mapping the same FilterOutcomes into an estimated
    P(graduate | observable state) using four covariates derived from the
    filters (graduation-curve progress, KOL participation, creator
    alignment, bot/cluster intensity). The mapping is grounded in the
    covariate signs / monotonicities reported in Marino, Naviglio,
    Tarantelli, Lillo (2026, arXiv:2602.14860).

Both produce a CompositeScore with `score` in [0, 100]; the orchestrator
treats them identically and compares against `ScoringConfig.enter_threshold`
(or the aggressive-profile override).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import structlog

from .config import ScoringConfig
from .models import CompositeScore, FilterOutcome

logger = structlog.get_logger(__name__)


def _outcome_map(outcomes: Iterable[FilterOutcome]) -> dict[str, FilterOutcome]:
    return {o.name: o for o in outcomes}


class CompositeScorer:
    """Classic weighted-sum scorer (v0.2 behaviour)."""

    def __init__(self, config: ScoringConfig) -> None:
        self.config = config

    def score(self, outcomes: Iterable[FilterOutcome]) -> CompositeScore:
        breakdown: dict[str, float] = {}
        hard_rejected = False
        reject_reason = ""
        weighted_total = 0.0

        items = list(outcomes)

        for outcome in items:
            breakdown[outcome.name] = outcome.score
            if outcome.hard_reject:
                hard_rejected = True
                if not reject_reason:
                    reject_reason = f"[{outcome.name}] {outcome.reject_reason}"

        if hard_rejected:
            return CompositeScore(
                score=0.0,
                breakdown=breakdown,
                hard_rejected=True,
                reject_reason=reject_reason,
                enter=False,
            )

        weights = self.config.weights.model_dump()
        for outcome in items:
            w = float(weights.get(outcome.name, 0.0))
            weighted_total += w * outcome.score

        score = max(0.0, min(100.0, weighted_total))
        enter = score >= self.config.enter_threshold

        return CompositeScore(
            score=score,
            breakdown=breakdown,
            hard_rejected=False,
            reject_reason="",
            enter=enter,
        )


# ── Lillo-Naviglio graduation-probability scorer (v0.3) ────────────────────


# Logistic-regression-style coefficients. These are *not* fit to a private
# dataset; they encode the qualitative covariate signs reported in
# Marino et al. 2026 and pump.fun's own public fee/creator-rewards docs.
# Magnitudes are set so the resulting probability spans the practically
# useful [0.02, 0.85] range across realistic filter outcomes (the paper's
# base rate is ~0.63%, the high-conviction tail ~30%+).
#
# These are deliberately exposed as class attributes (not config knobs) so
# the strategy is reproducible across runs; we can pin them in unit tests.


class GraduationProbabilityScorer:
    """Score = 100 * P(graduate | observable state).

    Inputs (each derived from a FilterOutcome):

        curve   = curve_graduation.score       (proxy for trade-accumulation
                  speed + distinct-buyer count + velocity)
        kol     = max(first_kol_touch.score,
                      convergence.score)       (smart-money participation)
        creator = creator_vault.score          (creator-pubkey alignment)
        bot     = 100 - bundle_cluster.score   (concentration ⇒ bot intensity)

    z = b0 + b_curve * curve + b_kol * kol + b_creator * creator + b_bot * bot
    P = sigmoid(z) ∈ (0, 1)
    score = 100 * P

    Signs:
      b_curve   > 0   (more curve progress ⇒ closer to graduation)
      b_kol     > 0   (KOL participation predicts continuation)
      b_creator > 0   (aligned creator increases P)
      b_bot     < 0   (high bot concentration depresses P; matches the
                        bundle-detector hard-reject rule)

    Magnitudes are tuned so:
      - neutral signal (all filters at 50, bundle=50)        ⇒ score ≈ 8 %
      - aggressive-profile entry surface (curve=60, kol=70,
        creator=80, bundle=70 → bot_conc=30)                  ⇒ score ≈ 48 %
      - conviction surface (curve=95, kol=100, creator=80,
        bundle=60 → bot_conc=40)                              ⇒ score ≈ 86 %
      - neutral except high bot concentration (bundle=20
        → bot_conc=80)                                        ⇒ score ≈  2 %

    The combined effect is that the v0.3 `enter_threshold=40` lines up with
    a real ~47 % probability of graduation — roughly 75x the unconditional
    base rate of 0.63 % reported in the Marino et al. dataset.
    """

    NAME = "graduation_probability"

    # Coefficients (per-point of filter score). Each filter is 0-100 so a
    # 0.04 coefficient produces +4 in z at maximum.
    B0: float = -5.0
    B_CURVE: float = 0.04
    B_KOL: float = 0.03
    B_CREATOR: float = 0.02
    B_BOT: float = -0.04

    def __init__(self, config: ScoringConfig) -> None:
        self.config = config

    @staticmethod
    def _sigmoid(z: float) -> float:
        # Clamp to avoid math.exp overflow on extreme inputs.
        if z >= 0:
            ez = math.exp(-z) if z < 50 else 0.0
            return 1.0 / (1.0 + ez)
        ez = math.exp(z) if z > -50 else 0.0
        return ez / (1.0 + ez)

    def score(self, outcomes: Iterable[FilterOutcome]) -> CompositeScore:
        items = list(outcomes)
        breakdown: dict[str, float] = {o.name: o.score for o in items}

        hard_rejected = False
        reject_reason = ""
        for o in items:
            if o.hard_reject:
                hard_rejected = True
                if not reject_reason:
                    reject_reason = f"[{o.name}] {o.reject_reason}"

        if hard_rejected:
            return CompositeScore(
                score=0.0,
                breakdown=breakdown,
                hard_rejected=True,
                reject_reason=reject_reason,
                enter=False,
            )

        m = _outcome_map(items)

        def _val(name: str, default: float = 50.0) -> float:
            o = m.get(name)
            return float(o.score) if o is not None else default

        curve = _val("curve_graduation")
        kol = max(_val("first_kol_touch"), _val("convergence"))
        creator = _val("creator_vault")
        # bundle_detector reports HIGH score when bundle concentration is LOW
        # (good); we invert so b_bot < 0 punishes high concentration.
        bot_concentration_score = 100.0 - _val("bundle_cluster")

        z = (
            self.B0
            + self.B_CURVE * curve
            + self.B_KOL * kol
            + self.B_CREATOR * creator
            + self.B_BOT * bot_concentration_score
        )
        probability = self._sigmoid(z)
        score_pct = max(0.0, min(100.0, 100.0 * probability))

        # Record the four-covariate decomposition for the dashboard.
        breakdown = dict(breakdown)
        breakdown["_grad_prob"] = score_pct
        breakdown["_z"] = z

        return CompositeScore(
            score=score_pct,
            breakdown=breakdown,
            hard_rejected=False,
            reject_reason="",
            enter=score_pct >= self.config.enter_threshold,
        )


# ── Factory ───────────────────────────────────────────────────────────────


def build_scorer(
    config: ScoringConfig,
) -> CompositeScorer | GraduationProbabilityScorer:
    """Build the scorer indicated by `config.mode`."""
    if config.mode == "graduation_probability":
        return GraduationProbabilityScorer(config)
    return CompositeScorer(config)
