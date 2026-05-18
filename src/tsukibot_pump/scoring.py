"""Composite scorer: combine 6 filter outcomes into a single 0-100 score.

Hard-reject semantics: if any filter hard-rejects, the composite is
hard-rejected (`enter = False`, `score = 0`) regardless of other scores.

Otherwise: weighted sum using `ScoringWeights`. `enter` is set to True iff
`score >= enter_threshold` (read from config by the orchestrator, not here).
"""

from __future__ import annotations

from collections.abc import Iterable

import structlog

from .config import ScoringConfig
from .models import CompositeScore, FilterOutcome

logger = structlog.get_logger(__name__)


class CompositeScorer:
    def __init__(self, config: ScoringConfig) -> None:
        self.config = config

    def score(self, outcomes: Iterable[FilterOutcome]) -> CompositeScore:
        breakdown: dict[str, float] = {}
        hard_rejected = False
        reject_reason = ""
        weighted_total = 0.0

        # Materialize once so we can iterate twice.
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
