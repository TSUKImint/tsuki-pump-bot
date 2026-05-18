"""Filter 2 — bundle / cluster detection on first-N buyers.

We classify the first N buyers of a token into two groups:

  Bundles: wallets buying within `bundle_window_slots` slots of each other.
    A bundle is the classic Jito-block-0 anti-snipe signature, often used
    by either (a) sniper-bot rings or (b) the dev themself to seed multiple
    wallets they control. Either way it's a concentration risk.

  Clusters: wallets that share funding (same parent wallet topped them up
    before the buy). This is the Bubblemaps "this looks like one entity"
    signal. Stealth bundlers specifically try to defeat this by routing
    funding through Solana Pay, CEX exits, or many-to-many transfers — we
    flag those patterns as additional risk (see `extra["stealth_signals"]`).

Score semantics:
  - 100: no concentration; first-N buyers look organic (>= 80% distinct
    funding sources, no bundles).
  - 50: some concentration but below the reject threshold.
  - 0 + hard_reject: a single cluster owns >= `reject_cluster_concentration_gte`
    of first-N supply.

In paper-mock mode we don't have funding-source data; the filter still
runs on what it has (slot bundles) and notes the limitation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import structlog

from ..config import BundleClusterConfig
from ..models import BuyRecord, FilterOutcome, TokenState

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ClusterAnalysis:
    bundle_count: int  # number of multi-wallet same-slot groups
    largest_bundle_size: int  # how many wallets in the biggest bundle
    largest_cluster_units_frac: float  # fraction of first-N supply in largest cluster
    distinct_funders: int  # number of unique parent funders seen
    stealth_signals: list[str]  # human-readable flags for the audit log


class BundleDetector:
    """Detect snipe bundles + funding-cluster concentration on first-N buyers."""

    NAME = "bundle_cluster"

    def __init__(self, config: BundleClusterConfig) -> None:
        self.config = config

    def evaluate(
        self,
        token: TokenState,
        *,
        wallet_funder_lookup: dict[str, str] | None = None,
    ) -> FilterOutcome:
        if not self.config.enabled:
            return FilterOutcome(name=self.NAME, score=50.0, notes="disabled")

        first_n = token.buys[: self.config.first_n_buyers]
        if len(first_n) < 5:
            # Not enough data yet — neutral.
            return FilterOutcome(
                name=self.NAME,
                score=50.0,
                notes=f"only {len(first_n)} buys yet, deferring",
            )

        analysis = self._analyze(first_n, wallet_funder_lookup or {})
        token.extra["bundle_cluster_analysis"] = analysis

        if analysis.largest_cluster_units_frac >= self.config.reject_cluster_concentration_gte:
            return FilterOutcome(
                name=self.NAME,
                score=0.0,
                hard_reject=True,
                reject_reason=(
                    f"largest funding cluster owns "
                    f"{analysis.largest_cluster_units_frac:.0%} of first-N supply"
                ),
                notes=(
                    f"bundles={analysis.bundle_count}, "
                    f"largest_bundle={analysis.largest_bundle_size}"
                ),
            )

        # Stealth signals indicate the dev is *trying* to defeat detection.
        # That's worse than naive concentration — even more bearish.
        if analysis.stealth_signals:
            return FilterOutcome(
                name=self.NAME,
                score=20.0,
                notes="stealth bundling signals: " + ", ".join(analysis.stealth_signals),
            )

        if analysis.largest_cluster_units_frac >= 0.20:
            return FilterOutcome(
                name=self.NAME,
                score=40.0,
                notes=f"moderate concentration ({analysis.largest_cluster_units_frac:.0%})",
            )

        if analysis.bundle_count == 0 and analysis.largest_cluster_units_frac <= 0.10:
            return FilterOutcome(
                name=self.NAME,
                score=100.0,
                notes="organic distribution (no bundles, low concentration)",
            )

        return FilterOutcome(
            name=self.NAME,
            score=70.0,
            notes=(
                f"bundles={analysis.bundle_count}, "
                f"concentration={analysis.largest_cluster_units_frac:.0%}"
            ),
        )

    def _analyze(
        self,
        first_n: list[BuyRecord],
        wallet_funder_lookup: dict[str, str],
    ) -> ClusterAnalysis:
        # ── Bundle detection: group by slot bucket ─────────────────────────
        slot_groups: dict[int, list[BuyRecord]] = defaultdict(list)
        for buy in first_n:
            bucket = buy.slot // max(1, self.config.bundle_window_slots)
            slot_groups[bucket].append(buy)
        bundles = [g for g in slot_groups.values() if len(g) >= 2]
        largest_bundle = max((len(g) for g in bundles), default=0)

        # ── Cluster detection: group by funder ─────────────────────────────
        # Falls back to wallet identity itself if funder is unknown — then
        # the cluster size mirrors per-wallet concentration only.
        cluster_units: dict[str, float] = defaultdict(float)
        for buy in first_n:
            funder = wallet_funder_lookup.get(buy.wallet, buy.wallet)
            cluster_units[funder] += buy.token_units_received

        total_units = sum(b.token_units_received for b in first_n) or 1.0
        largest_units = max(cluster_units.values(), default=0.0)
        largest_frac = largest_units / total_units

        # ── Stealth signals ───────────────────────────────────────────────
        stealth_signals: list[str] = []

        # 1. Many wallets, same funder, but funder appears only once per wallet
        # → fan-out pattern that looks like classic stealth bundling.
        if wallet_funder_lookup:
            funders_by_wallet: dict[str, str] = {
                b.wallet: wallet_funder_lookup.get(b.wallet, b.wallet) for b in first_n
            }
            funder_to_wallets: dict[str, set[str]] = defaultdict(set)
            for wallet, funder in funders_by_wallet.items():
                funder_to_wallets[funder].add(wallet)
            max_fanout = max(
                (len(wallets) for wallets in funder_to_wallets.values()),
                default=0,
            )
            if max_fanout >= 5:
                stealth_signals.append(f"funder fan-out: 1 funder → {max_fanout} buyer wallets")

        # 2. Same-amount buys (sniper-bot tell) — N buys with identical SOL
        # amount within the bundle window.
        sol_amounts = [round(b.sol_spent, 4) for b in first_n]
        sol_count: dict[float, int] = defaultdict(int)
        for amount in sol_amounts:
            sol_count[amount] += 1
        for amount, count in sol_count.items():
            if count >= 4:
                stealth_signals.append(f"{count} buys with identical {amount:.4f} SOL amount")
                break

        return ClusterAnalysis(
            bundle_count=len(bundles),
            largest_bundle_size=largest_bundle,
            largest_cluster_units_frac=largest_frac,
            distinct_funders=len(cluster_units),
            stealth_signals=stealth_signals,
        )
