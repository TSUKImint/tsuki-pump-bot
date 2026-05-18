"""Filter 3 — first-KOL-touch signal.

Backtested over 491,000 KOL trades by MadeOnSol (2026):
"a tracked KOL was first to buy this token" *is* a real signal, but only
conditional on which KOL. Generic copy-trade isn't profitable. Top-decile
KOL first-touch is.

This filter reads a private KOL leaderboard (CSV: wallet, label, score)
and reports how many of the first-N buyers are on the list. The "secret"
KOL list quality IS the edge — wallets that appear in public Cielo /
Nansen dashboards are already being front-run.

Score semantics:
  - score_per_touch points per matched KOL, clamped at max_kol_touches.
  - A boost of +20% if at least one KOL is in the very first 3 buys
    (signals high conviction, not just opportunistic follow-on).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import structlog

from ..config import FirstKolTouchConfig
from ..models import FilterOutcome, TokenState

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class KolEntry:
    wallet: str
    label: str
    score: float  # 0-100 weight; higher = better KOL


class FirstKolTouch:
    """Score by how many tracked KOLs were among the first-N buyers."""

    NAME = "first_kol_touch"

    def __init__(self, config: FirstKolTouchConfig) -> None:
        self.config = config
        self._kols: dict[str, KolEntry] = self._load(config.kol_csv_path)
        logger.info(
            "first_kol_touch.loaded",
            entries=len(self._kols),
            path=str(config.kol_csv_path),
        )

    @staticmethod
    def _load(path: Path) -> dict[str, KolEntry]:
        """Load KOL list from CSV: wallet,label,score (or wallet,label)."""
        if not path.exists():
            logger.warning(
                "first_kol_touch.file_missing",
                path=str(path),
                note="filter will always return neutral score",
            )
            return {}
        out: dict[str, KolEntry] = {}
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                wallet = row[0].strip()
                if not wallet:
                    continue
                label = row[1].strip() if len(row) > 1 else ""
                try:
                    score = float(row[2]) if len(row) > 2 else 50.0
                except ValueError:
                    score = 50.0
                out[wallet] = KolEntry(wallet=wallet, label=label, score=score)
        return out

    def evaluate(self, token: TokenState) -> FilterOutcome:
        if not self.config.enabled:
            return FilterOutcome(name=self.NAME, score=50.0, notes="disabled")
        if not self._kols:
            return FilterOutcome(name=self.NAME, score=50.0, notes="no KOL list loaded")

        first_n = token.buys[: self.config.first_n_buyers]
        if not first_n:
            return FilterOutcome(name=self.NAME, score=50.0, notes="no buys yet")

        matched: list[tuple[int, KolEntry]] = []
        for idx, buy in enumerate(first_n):
            kol = self._kols.get(buy.wallet)
            if kol is not None:
                matched.append((idx, kol))

        token.extra["kol_matches"] = [(idx, k.label or k.wallet[:8]) for idx, k in matched]

        if not matched:
            return FilterOutcome(name=self.NAME, score=30.0, notes="no KOL among first-N buyers")

        touches = min(len(matched), self.config.max_kol_touches)
        base_score = min(100.0, touches * self.config.score_per_touch)

        # Bonus for first-3 placement (conviction signal).
        if any(idx < 3 for idx, _ in matched):
            base_score = min(100.0, base_score * 1.2)

        # Weight by average KOL score (top-decile KOL > middling).
        avg_kol_score = sum(k.score for _, k in matched) / len(matched)
        scaled = base_score * (avg_kol_score / 100.0)

        labels = ", ".join(k.label or k.wallet[:8] for _, k in matched[:3])
        return FilterOutcome(
            name=self.NAME,
            score=min(100.0, scaled),
            notes=f"{len(matched)} KOL touches (avg score {avg_kol_score:.0f}): {labels}",
        )
