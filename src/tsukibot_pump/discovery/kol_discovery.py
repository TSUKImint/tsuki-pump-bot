"""KOL discovery orchestrator — turns a history into a scored leaderboard.

The flow is intentionally linear so each step is easy to test in
isolation:

    1. classify_token_outcomes(events) -> {mint: TokenOutcome}
    2. build_wallet_features(events, outcomes) -> {wallet: WalletFeatures}
    3. For each wallet:
       a. Apply recency cut (last trade within N days; >= M recent trades).
       b. Apply min-sample-size cut (need >= K decided trades to score).
       c. Run bot heuristics (sniper / mechanical / high-frequency).
       d. Run poison heuristics (funder graph / dump-bus).
       e. Compute score from hit_rate, log_roi_mean, recency, n_trades.
    4. Sort by score desc, truncate to top-N, return.

Outputs `DiscoveredKol` records that can be persisted as
``data/private_kol_list.csv`` via :func:`write_kol_csv` — the format
matches what ``first_kol_touch`` and ``convergence`` already consume.

This module is pure-functional: no RPC calls, no disk reads. Callers
gather the data however they prefer (firehose recording, event store
backfill, third-party CSV) and pass it in.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from ..solana.pump_program import PumpEvent
from .bot_heuristics import BotHeuristicsConfig, is_likely_bot
from .outcomes import (
    TokenOutcome,
    TokenOutcomeLabel,
    classify_token_outcomes,
    is_positive_outcome,
)
from .poison_heuristics import PoisonHeuristicsConfig, is_likely_poison_wallet
from .wallet_features import WalletFeatures, build_wallet_features

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class KolDiscoveryConfig:
    # Outcome classification.
    min_multiple_winner: float = 3.0
    graduation_real_sol_threshold: float = 85.0
    observation_window_seconds: int = 24 * 3600

    # Recency / minimum-activity cuts (the user's hard requirement —
    # don't anoint abandoned wallets).
    require_last_trade_within_seconds: int = 7 * 24 * 3600  # 1 week
    require_n_trades_within_seconds: int = 14 * 24 * 3600  # 2 weeks
    min_trades_in_recency_window: int = 3
    # Drop wallets with fewer than this many decided trades total — the
    # hit rate is too noisy below this.
    min_decided_trades: int = 5

    # Scoring weights — sum doesn't have to be 1, the final score is
    # normalised to [0, 100] by clamping the raw weighted sum.
    weight_hit_rate: float = 40.0
    weight_log_roi: float = 30.0
    weight_recency: float = 15.0
    weight_volume: float = 15.0
    # log_roi gets clipped at this multiple to prevent a single 1000x
    # from dominating. 3.5 in log ≈ 33x — generous but not unbounded.
    log_roi_clip: float = 3.5

    # Top-N output.
    top_n: int = 200

    # Sub-configs.
    bot_heuristics: BotHeuristicsConfig = field(default_factory=BotHeuristicsConfig)
    poison_heuristics: PoisonHeuristicsConfig = field(default_factory=PoisonHeuristicsConfig)


@dataclass(frozen=True, slots=True)
class DiscoveredKol:
    wallet: str
    label: str  # CSV "label" column; we put a short rationale here
    score: float  # 0..100
    hit_rate: float
    log_roi_mean: float
    n_decided_trades: int
    last_seen_unix: int | None
    # Diagnostic fields (not written to CSV but useful for reports).
    rejected: bool = False
    rejection_reason: str = ""


def discover_kols(
    events: Iterable[PumpEvent],
    *,
    config: KolDiscoveryConfig | None = None,
    funder_lookup: Mapping[str, str] | None = None,
    sell_destination_lookup: Mapping[str, list[str]] | None = None,
    now_unix: int | None = None,
) -> list[DiscoveredKol]:
    """Score wallets and return the top-N as ``DiscoveredKol`` records.

    ``funder_lookup`` and ``sell_destination_lookup`` are optional — if
    not provided, the poison heuristics fall back to "shared-funder
    fraction = 0" and skip the dump-bus check. That's a strictly weaker
    filter but lets the module work on partial data.
    """
    cfg = config or KolDiscoveryConfig()
    # Materialise events once — we walk them twice (outcomes then features).
    events_list = list(events)

    outcomes = classify_token_outcomes(
        events_list,
        min_multiple_winner=cfg.min_multiple_winner,
        graduation_real_sol_threshold=cfg.graduation_real_sol_threshold,
        observation_window_seconds=cfg.observation_window_seconds,
        now_unix=now_unix,
    )
    feats = build_wallet_features(events_list, outcomes)

    results: list[DiscoveredKol] = []
    for wallet, feat in feats.items():
        verdict = _score_one_wallet(
            wallet=wallet,
            feat=feat,
            outcomes=outcomes,
            cfg=cfg,
            funder_lookup=funder_lookup,
            sell_destination_lookup=sell_destination_lookup,
            now_unix=now_unix,
        )
        results.append(verdict)

    accepted = [r for r in results if not r.rejected]
    accepted.sort(key=lambda r: r.score, reverse=True)
    truncated = accepted[: cfg.top_n]
    logger.info(
        "kol_discovery.complete",
        wallets_examined=len(results),
        accepted=len(accepted),
        rejected=len(results) - len(accepted),
        emitted=len(truncated),
    )
    return truncated


def _score_one_wallet(
    *,
    wallet: str,
    feat: WalletFeatures,
    outcomes: Mapping[str, TokenOutcome],
    cfg: KolDiscoveryConfig,
    funder_lookup: Mapping[str, str] | None,
    sell_destination_lookup: Mapping[str, list[str]] | None,
    now_unix: int | None,
) -> DiscoveredKol:
    """Run the full reject-then-score pipeline for a single wallet."""

    # ── 1) Hard minimum activity ─────────────────────────────────────────
    if feat._decided_trades < cfg.min_decided_trades:
        return _reject(feat, f"only {feat._decided_trades} decided trades")

    # ── 2) Recency cut ───────────────────────────────────────────────────
    if now_unix is not None and feat.last_seen_unix is not None:
        seconds_since_last = now_unix - feat.last_seen_unix
        if seconds_since_last > cfg.require_last_trade_within_seconds:
            return _reject(
                feat,
                f"abandoned: last trade {seconds_since_last // 86_400}d ago",
            )

    if now_unix is not None and not _has_recent_activity(feat, cfg, now_unix):
        return _reject(
            feat,
            f"<{cfg.min_trades_in_recency_window} trades in last "
            f"{cfg.require_n_trades_within_seconds // 86_400}d",
        )

    # ── 3) Bot heuristics ────────────────────────────────────────────────
    is_bot, bot_reason = is_likely_bot(feat, cfg.bot_heuristics)
    if is_bot:
        return _reject(feat, f"bot-like: {bot_reason}")

    # ── 4) Poison heuristics ─────────────────────────────────────────────
    positive_outcomes = [
        outcomes[mint] for mint in _winning_mints_for_wallet(feat, outcomes) if mint in outcomes
    ]
    is_poison, poison_reason = is_likely_poison_wallet(
        wallet,
        positive_outcomes,
        config=cfg.poison_heuristics,
        funder_lookup=funder_lookup,
        sell_destination_lookup=sell_destination_lookup,
    )
    if is_poison:
        return _reject(feat, f"poison-like: {poison_reason}")

    # ── 5) Score ─────────────────────────────────────────────────────────
    score, label = _compute_score_and_label(feat, cfg, now_unix)
    return DiscoveredKol(
        wallet=wallet,
        label=label,
        score=score,
        hit_rate=feat.hit_rate,
        log_roi_mean=feat.log_roi_mean,
        n_decided_trades=feat._decided_trades,
        last_seen_unix=feat.last_seen_unix,
    )


def _reject(feat: WalletFeatures, reason: str) -> DiscoveredKol:
    return DiscoveredKol(
        wallet=feat.wallet,
        label="",
        score=0.0,
        hit_rate=feat.hit_rate,
        log_roi_mean=feat.log_roi_mean,
        n_decided_trades=feat._decided_trades,
        last_seen_unix=feat.last_seen_unix,
        rejected=True,
        rejection_reason=reason,
    )


def _has_recent_activity(feat: WalletFeatures, cfg: KolDiscoveryConfig, now_unix: int) -> bool:
    """We can only check this exactly with a per-trade timestamp scan, but
    we have a cheap proxy: trades_per_day projected over the recency
    window. Good enough as a guardrail — the funder/recency lookup paths
    can refine it later.
    """
    if feat.last_seen_unix is None or feat.first_seen_unix is None:
        return False
    span_seconds = max(1, feat.last_seen_unix - feat.first_seen_unix)
    rate = feat.n_observed_buys / span_seconds  # buys per second
    projected_in_window = rate * cfg.require_n_trades_within_seconds
    return projected_in_window >= cfg.min_trades_in_recency_window


def _winning_mints_for_wallet(
    feat: WalletFeatures, outcomes: Mapping[str, TokenOutcome]
) -> list[str]:
    """We don't store the per-wallet mint list explicitly in features
    (memory), so reconstruct by intersecting wallet feature counts. We
    only need this when poison checks fire (>= 3 positive trades), so
    it's not on the hot path.

    Cheap path: any outcome where the wallet appears in
    ``early_buyer_wallets``.
    """
    out: list[str] = []
    for mint, outcome in outcomes.items():
        if not is_positive_outcome(outcome.label):
            continue
        if feat.wallet in outcome.early_buyer_wallets:
            out.append(mint)
    return out


def _compute_score_and_label(
    feat: WalletFeatures,
    cfg: KolDiscoveryConfig,
    now_unix: int | None,
) -> tuple[float, str]:
    """Final scoring. Clamps to [0, 100]."""
    # Hit rate component → 0..weight_hit_rate (linear 0..1 hit rate).
    hit_component = cfg.weight_hit_rate * max(0.0, min(1.0, feat.hit_rate))

    # Log-ROI component → clip to log_roi_clip, scale to 0..weight_log_roi.
    clipped_log_roi = max(0.0, min(cfg.log_roi_clip, feat.log_roi_mean))
    log_roi_component = (
        cfg.weight_log_roi * (clipped_log_roi / cfg.log_roi_clip) if cfg.log_roi_clip > 0 else 0.0
    )

    # Recency component → exponential decay with half-life = window/2.
    recency_component = 0.0
    if now_unix is not None and feat.last_seen_unix is not None:
        age_seconds = max(0, now_unix - feat.last_seen_unix)
        half_life = max(1, cfg.require_last_trade_within_seconds // 2)
        recency_component = cfg.weight_recency * (0.5 ** (age_seconds / half_life))

    # Volume component → log-scaled n_decided_trades, capped at 10.
    volume_component = cfg.weight_volume * min(1.0, math.log10(feat._decided_trades + 1) / 1.0)

    raw = hit_component + log_roi_component + recency_component + volume_component
    score = max(0.0, min(100.0, raw))

    label = f"hit={feat.hit_rate:.0%} logROI={feat.log_roi_mean:.2f} n={feat._decided_trades}"
    return score, label


def write_kol_csv(kols: Iterable[DiscoveredKol], path: Path) -> int:
    """Write a leaderboard to a CSV in the format ``first_kol_touch`` expects.

    Format: ``wallet,label,score`` (one entry per line, ``#`` for comments).
    Returns the number of rows written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated by tsuki_pump.discovery.kol_discovery\n")
        f.write("# Format: wallet,label,score\n")
        f.write("# Wallets here have passed bot-detection, poison-detection,\n")
        f.write("# and recency checks. Review before trusting blindly.\n")
        for k in kols:
            if k.rejected:
                continue
            # Escape commas in label just in case.
            safe_label = k.label.replace(",", " ").strip()
            f.write(f"{k.wallet},{safe_label},{k.score:.2f}\n")
            n += 1
    return n


__all__ = [
    "DiscoveredKol",
    "KolDiscoveryConfig",
    "TokenOutcomeLabel",  # re-exported for typing convenience
    "discover_kols",
    "write_kol_csv",
]
