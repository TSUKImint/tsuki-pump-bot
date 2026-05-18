"""Filter 6 — Community-Takeover (CTO) detector.

Per Solana Radar (Feb 2026 coverage) and Coinlive's "MEME hype rollback"
piece, CTO is the dominant late-stage memecoin play in 2026. Examples:
$MICHI ($320M peak), $WIF, $POPCAT, $1DOL — all abandoned by their devs,
then revived by communities, then 10-100x'd.

The signal is the least speed-sensitive edge on the whole market: CTO
momentum builds over days to weeks, well within a laptop's reach.

Detection:
  - Dev wallet has been silent for >= `dev_silent_min_days`.
  - Token did NOT dump to zero (still has measurable activity).
  - Unique daily buyer count is >= `min_unique_buyers_24h`.
  - Token has been alive for >= `min_days_since_launch`.

Score semantics:
  - All criteria met → score 100 (rare, high-conviction CTO).
  - 3 of 4 criteria → score 60.
  - Fewer → score 40 (or score 30 if token is brand new).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

from ..config import CtoRevivalConfig
from ..models import FilterOutcome, TokenState

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CtoCriteria:
    dev_silent: bool
    not_dumped: bool
    active_community: bool
    old_enough: bool

    def all_met(self) -> bool:
        return all([self.dev_silent, self.not_dumped, self.active_community, self.old_enough])

    def count_met(self) -> int:
        return sum([self.dev_silent, self.not_dumped, self.active_community, self.old_enough])


class CtoDetector:
    """Score tokens that look like community-takeover plays."""

    NAME = "cto_revival"

    def __init__(self, config: CtoRevivalConfig) -> None:
        self.config = config

    def evaluate(
        self,
        token: TokenState,
        *,
        now_unix: int | None = None,
        unique_buyers_24h: int = 0,
    ) -> FilterOutcome:
        if not self.config.enabled:
            return FilterOutcome(name=self.NAME, score=50.0, notes="disabled")

        now = now_unix if now_unix is not None else int(datetime.now(tz=UTC).timestamp())
        created = token.created_at_unix
        if created is None:
            return FilterOutcome(name=self.NAME, score=30.0, notes="unknown launch age")

        age_seconds = max(0, now - created)
        age_days = age_seconds / 86400.0
        if age_days < self.config.min_days_since_launch:
            return FilterOutcome(
                name=self.NAME,
                score=30.0,
                notes=f"too fresh ({age_days:.1f}d < {self.config.min_days_since_launch}d)",
            )

        dev_last = token.dev_activity.last_seen_unix
        if dev_last is None:
            dev_silent = True  # never seen dev wallet activity → treat as silent
        else:
            dev_silent_seconds = max(0, now - dev_last)
            dev_silent = dev_silent_seconds >= self.config.dev_silent_min_days * 86400

        # "Not dumped" is a soft signal: token still has buys in recent
        # history. We use `distinct_buyers_60s` as a proxy proximate to
        # actual buyer count — a more thorough impl would query
        # `unique_buyers_24h`.
        not_dumped = unique_buyers_24h > 0 or token.distinct_buyers_60s > 0
        active_community = unique_buyers_24h >= self.config.min_unique_buyers_24h
        old_enough = age_days >= self.config.min_days_since_launch

        criteria = CtoCriteria(
            dev_silent=dev_silent,
            not_dumped=not_dumped,
            active_community=active_community,
            old_enough=old_enough,
        )

        if criteria.all_met():
            return FilterOutcome(
                name=self.NAME,
                score=100.0,
                notes=(
                    f"CTO candidate: dev silent {dev_silent_seconds // 86400}d, "
                    f"{unique_buyers_24h} unique buyers/24h"
                    if dev_last
                    else f"CTO candidate: dev never seen, {unique_buyers_24h} unique buyers/24h"
                ),
            )

        n = criteria.count_met()
        if n == 3:
            return FilterOutcome(name=self.NAME, score=60.0, notes="3/4 CTO criteria met")
        if n == 2:
            return FilterOutcome(name=self.NAME, score=40.0, notes="2/4 CTO criteria met")
        return FilterOutcome(name=self.NAME, score=30.0, notes=f"{n}/4 CTO criteria met")
