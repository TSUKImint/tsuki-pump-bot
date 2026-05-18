"""Filter 4 — KOL convergence.

Sister-filter to first-KOL-touch: instead of "any KOL was first", we look
for "two or more tracked KOLs bought the same token within a short window".
That joint signal is materially stronger than a single KOL buy.

XHuntr's "convergence alert" pattern (xhuntr.com docs) is the public face
of this. The edge decays as more retail copy-traders also watch convergence,
so the secret-KOL list quality remains the moat.

Score semantics:
  - At least `min_distinct_kols` distinct KOLs within `window_seconds` →
    `score_at_minimum`.
  - Each extra KOL above the minimum: `score_per_extra_kol`, capped at 100.
  - No convergence (or only 1 KOL): score = 40 (neutral-low).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import structlog

from ..config import ConvergenceConfig, FirstKolTouchConfig
from ..models import FilterOutcome, TokenState

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class KolEntry:
    wallet: str
    label: str
    score: float


class ConvergenceDetector:
    """Boost score when N+ KOLs converge on the same token in a short window."""

    NAME = "convergence"

    def __init__(
        self,
        config: ConvergenceConfig,
        *,
        kol_csv_path: Path,
    ) -> None:
        self.config = config
        self._kols = self._load(kol_csv_path)
        logger.info(
            "convergence.loaded",
            entries=len(self._kols),
            path=str(kol_csv_path),
        )

    @staticmethod
    def _load(path: Path) -> dict[str, KolEntry]:
        if not path.exists():
            logger.warning("convergence.file_missing", path=str(path))
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

    @classmethod
    def from_kol_filter_config(
        cls,
        convergence_config: ConvergenceConfig,
        kol_filter_config: FirstKolTouchConfig,
    ) -> ConvergenceDetector:
        """Convenience: reuse the same KOL CSV the first-touch filter uses."""
        return cls(convergence_config, kol_csv_path=kol_filter_config.kol_csv_path)

    def evaluate(self, token: TokenState) -> FilterOutcome:
        if not self.config.enabled:
            return FilterOutcome(name=self.NAME, score=50.0, notes="disabled")
        if not self._kols:
            return FilterOutcome(name=self.NAME, score=50.0, notes="no KOL list loaded")
        if not token.buys:
            return FilterOutcome(name=self.NAME, score=40.0, notes="no buys yet")

        # Collect KOL buys with timestamps; we use block_time_unix where
        # available, otherwise slot/2 as a rough proxy.
        kol_buys: list[tuple[int, KolEntry]] = []
        for buy in token.buys:
            kol = self._kols.get(buy.wallet)
            if kol is None:
                continue
            ts = buy.block_time_unix
            if ts is None:
                ts = buy.slot // 2  # rough estimate
            kol_buys.append((ts, kol))

        if len(kol_buys) < self.config.min_distinct_kols:
            return FilterOutcome(
                name=self.NAME,
                score=40.0,
                notes=f"only {len(kol_buys)} KOL buys (need {self.config.min_distinct_kols})",
            )

        # Sort by timestamp and slide a window.
        kol_buys.sort(key=lambda x: x[0])
        best_window: list[KolEntry] = []
        for i in range(len(kol_buys)):
            j = i
            distinct: set[str] = set()
            current: list[KolEntry] = []
            window_end = kol_buys[i][0] + self.config.window_seconds
            while j < len(kol_buys) and kol_buys[j][0] <= window_end:
                if kol_buys[j][1].wallet not in distinct:
                    distinct.add(kol_buys[j][1].wallet)
                    current.append(kol_buys[j][1])
                j += 1
            if len(current) > len(best_window):
                best_window = current

        n = len(best_window)
        if n < self.config.min_distinct_kols:
            return FilterOutcome(
                name=self.NAME,
                score=40.0,
                notes=f"max simultaneous KOLs in window = {n}",
            )

        score = self.config.score_at_minimum + (
            (n - self.config.min_distinct_kols) * self.config.score_per_extra_kol
        )
        # Weight by average KOL score in the converging set.
        avg_kol_score = sum(k.score for k in best_window) / n
        scaled = min(100.0, score * (avg_kol_score / 100.0))

        labels = ", ".join(k.label or k.wallet[:8] for k in best_window[:3])
        return FilterOutcome(
            name=self.NAME,
            score=scaled,
            notes=f"{n} KOLs converged within {self.config.window_seconds}s: {labels}",
        )
