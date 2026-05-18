"""Filter 1 — dev wallet reputation blacklist.

Highest-impact, cheapest filter in the stack. Per public scammer-wallet
databases (AllenHark, WalletMaster, DeFade), 40-60% of new pump.fun tokens
are launched by repeat ruggers, and the same 4,000+ wallets re-rug
repeatedly. A simple blacklist + recent-token-velocity heuristic catches
the majority of these.

Operation:
  - Load CSV of known scammer wallets at construction time.
  - On each token, if `dev_wallet` is in the blacklist → hard-reject.
  - Optionally count recent launches by `dev_wallet` from our own
    event store (tokens table) and reject if velocity is suspicious.

Score semantics:
  - Hard reject (score = 0, hard_reject = True): known rugger, OR too many
    recent launches.
  - Score 100: dev wallet not seen in blacklist AND has never launched
    a token before in our event store (first-launch dev).
  - Score 50: in between (some prior launches but none flagged).
"""

from __future__ import annotations

import csv
from pathlib import Path

import structlog

from ..config import DevBlacklistConfig
from ..models import FilterOutcome, TokenState

logger = structlog.get_logger(__name__)


class DevBlacklist:
    """Reputation filter — rejects known ruggers + serial launchers."""

    NAME = "dev_blacklist"

    def __init__(
        self,
        config: DevBlacklistConfig,
        *,
        dev_token_count_24h_provider: object | None = None,
        dev_token_count_7d_provider: object | None = None,
    ) -> None:
        self.config = config
        # The provider callbacks let the orchestrator inject the event-store
        # query without this module having to know about SQLite.
        self._count_24h = dev_token_count_24h_provider
        self._count_7d = dev_token_count_7d_provider
        self._blacklist: frozenset[str] = self._load_blacklist(config.blacklist_csv_path)
        logger.info(
            "dev_blacklist.loaded",
            entries=len(self._blacklist),
            path=str(config.blacklist_csv_path),
        )

    @staticmethod
    def _load_blacklist(path: Path) -> frozenset[str]:
        """Load known-rugger wallet addresses from a CSV file.

        File format: one base58 pubkey per line (optional `,reason` after a
        comma; we ignore the rest). Missing file = empty set (warn).
        """
        if not path.exists():
            logger.warning(
                "dev_blacklist.file_missing",
                path=str(path),
                note="filter degrades to count-based only",
            )
            return frozenset()
        entries: set[str] = set()
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                wallet = row[0].strip()
                if not wallet or wallet.startswith("#"):
                    continue
                entries.add(wallet)
        return frozenset(entries)

    def evaluate(
        self,
        token: TokenState,
        *,
        dev_token_count_24h: int = 0,
        dev_token_count_7d: int = 0,
    ) -> FilterOutcome:
        if not self.config.enabled:
            return FilterOutcome(name=self.NAME, score=50.0, notes="disabled")

        dev = token.dev_wallet or ""
        if not dev:
            return FilterOutcome(
                name=self.NAME,
                score=40.0,
                notes="unknown dev wallet (suspicious, no proof of identity yet)",
            )

        if self.config.reject_if_known_rugger and dev in self._blacklist:
            return FilterOutcome(
                name=self.NAME,
                score=0.0,
                hard_reject=True,
                reject_reason=f"dev wallet {dev[:8]}... on rugger blacklist",
                notes="known scammer",
            )

        if dev_token_count_24h >= self.config.reject_if_dev_token_count_24h_gte:
            return FilterOutcome(
                name=self.NAME,
                score=0.0,
                hard_reject=True,
                reject_reason=f"dev launched {dev_token_count_24h} tokens in 24h",
                notes="serial launcher (24h velocity)",
            )

        if dev_token_count_7d >= self.config.reject_if_dev_token_count_7d_gte:
            return FilterOutcome(
                name=self.NAME,
                score=0.0,
                hard_reject=True,
                reject_reason=f"dev launched {dev_token_count_7d} tokens in 7d",
                notes="serial launcher (7d velocity)",
            )

        # Reward first-launch devs; penalise busy ones below the hard
        # thresholds.
        if dev_token_count_7d == 0:
            return FilterOutcome(name=self.NAME, score=100.0, notes="first-launch dev")
        if dev_token_count_7d <= 2:
            return FilterOutcome(
                name=self.NAME,
                score=70.0,
                notes=f"{dev_token_count_7d} prior launches in 7d",
            )
        return FilterOutcome(
            name=self.NAME,
            score=40.0,
            notes=f"{dev_token_count_7d} prior launches in 7d (warning)",
        )
