"""Performance report core.

Pure-functional analysis layer over the rows produced by
:class:`tsukibot_pump.core.event_store.EventStore`. The reporter does
not own the DB connection — callers fetch rows however they want
(``EventStore.recent_events``, raw SQL, etc.) and pass them in.

The report answers the four questions you actually care about when
deciding whether the bot has edge:

1. **Did closed paper positions make money?** Win rate, total P&L,
   median trade P&L. No survivor bias — closed positions only.
2. **Where did the spread go?** Aggregate pump.fun fees, priority
   fees, and latency-drift cost, surfaced from the diagnostic fields
   added by the paper-realism executor.
3. **Does the composite score predict P&L?** Bucket closed positions
   by ``score_at_entry`` and show win rate + median P&L per bucket.
   If high-score and low-score buckets have similar P&L, the score
   has no edge and the bot is just gambling.
4. **Why were exits triggered?** Distribution of exit reasons
   (``stop``, ``take_profit_*``, ``dev_drained``, etc.) and average
   P&L per reason. If ``stop`` dominates and TP rarely fires, the
   exit ladder is wrong.

Plus operational counters: events by kind (number of buys / sells /
rejects / failures), so you can spot e.g. high ``paper.buy_failed``
rates without grepping the audit log.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any


@dataclass(frozen=True, slots=True)
class EventCounts:
    """Count of events grouped by ``kind``."""

    total: int
    by_kind: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class FeeBreakdown:
    """Aggregate of the diagnostic fields written by the realism executor."""

    pump_fee_sol_total: float
    priority_fee_sol_total: float
    drift_sol_absorbed_total: float
    fills_observed: int  # how many fills contributed to these totals

    @property
    def total_friction_sol(self) -> float:
        """Sum of all paper friction costs — what mainnet would have cost extra."""
        return self.pump_fee_sol_total + self.priority_fee_sol_total + self.drift_sol_absorbed_total


@dataclass(frozen=True, slots=True)
class LatencyStats:
    p50_ms: float
    p90_ms: float
    p99_ms: float
    max_ms: float
    samples: int


@dataclass(frozen=True, slots=True)
class BucketStats:
    """Closed-position stats for one cohort (e.g. score bucket)."""

    n_positions: int
    n_winners: int  # realized_pnl_sol > 0
    win_rate: float  # n_winners / n_positions, 0 if no positions
    median_pnl_sol: float
    mean_pnl_sol: float
    total_pnl_sol: float
    # Sharpe-like risk-adjusted return: mean / stdev. Reported as None when
    # we have fewer than 2 positions (stdev undefined) or when stdev is 0.
    pnl_sharpe: float | None


@dataclass(frozen=True, slots=True)
class ScoreBucket:
    """One ``score_at_entry`` cohort plus its stats."""

    label: str
    score_min: float
    score_max: float
    stats: BucketStats


@dataclass(frozen=True, slots=True)
class ExitReasonStats:
    reason: str
    n_exits: int
    total_pnl_sol: float
    mean_pnl_sol: float


@dataclass(frozen=True, slots=True)
class AggregateReport:
    """Top-level container for everything :func:`format_report_text` renders."""

    event_counts: EventCounts
    overall: BucketStats  # all closed positions, regardless of bucket
    score_buckets: tuple[ScoreBucket, ...]
    exit_reasons: tuple[ExitReasonStats, ...]
    fees: FeeBreakdown
    latency: LatencyStats | None
    reject_reasons: Mapping[str, int] = field(default_factory=dict)
    paper_fail_buy: int = 0
    paper_fail_sell: int = 0


# ── public API ───────────────────────────────────────────────────────────


def build_report(
    *,
    events: Iterable[Mapping[str, Any]],
    positions: Iterable[Mapping[str, Any]],
    score_bucket_edges: tuple[float, ...] = (0.0, 40.0, 60.0, 80.0, 100.001),
) -> AggregateReport:
    """Compute an :class:`AggregateReport` from already-loaded rows.

    ``events`` should be the full ``events`` table (or a filtered view —
    e.g. only the last 24 hours), and ``positions`` should be the
    ``positions`` table. Each row is a ``Mapping`` keyed by column name,
    matching what ``EventStore`` already returns from its
    ``recent_events`` / ``list_open_positions`` helpers.

    ``score_bucket_edges`` defaults to four buckets aligned with the
    bot's composite score range: <40 / 40-60 / 60-80 / 80+. The
    rightmost edge is just above 100 to make the interval right-open.
    """
    events_list = list(events)
    positions_list = list(positions)

    counts_by_kind: Counter[str] = Counter()
    reject_reasons: Counter[str] = Counter()
    paper_fail_buy = 0
    paper_fail_sell = 0
    pump_fee_total = 0.0
    priority_fee_total = 0.0
    drift_total = 0.0
    latency_samples: list[float] = []
    n_fills = 0

    for ev in events_list:
        kind = str(ev.get("kind") or "")
        counts_by_kind[kind] += 1

        # Failures don't carry a fill payload, only buys + sells do.
        if kind == "paper.buy_failed":
            paper_fail_buy += 1
            continue
        if kind == "paper.sell_failed":
            paper_fail_sell += 1
            continue
        if kind == "risk.reject":
            summary = str(ev.get("summary") or "")
            # The orchestrator format is `refused buy for <mint8>: <reason>`.
            reason_part = summary.split(":", 1)[1].strip() if ":" in summary else summary
            if reason_part:
                reject_reasons[reason_part] += 1
            continue

        if kind not in {"paper.buy", "paper.sell"}:
            continue

        payload = _load_payload(ev)
        fill = payload.get("fill") if isinstance(payload, dict) else None
        if not isinstance(fill, dict):
            continue
        pump_fee_total += _as_float(fill.get("pump_fee_sol"))
        priority_fee_total += _as_float(fill.get("priority_fee_sol"))
        drift_total += _as_float(fill.get("drift_sol_absorbed"))
        latency = fill.get("latency_ms")
        if latency is not None:
            latency_samples.append(_as_float(latency))
        n_fills += 1

    event_counts = EventCounts(total=len(events_list), by_kind=dict(counts_by_kind))
    fees = FeeBreakdown(
        pump_fee_sol_total=pump_fee_total,
        priority_fee_sol_total=priority_fee_total,
        drift_sol_absorbed_total=drift_total,
        fills_observed=n_fills,
    )
    latency = _summarise_latency(latency_samples)

    # Positions side — only closed ones with a non-null PnL contribute.
    closed_positions = [
        p
        for p in positions_list
        if str(p.get("status") or "") == "closed" and p.get("realized_pnl_sol") is not None
    ]
    overall = _bucket_stats(closed_positions)
    score_buckets = _build_score_buckets(closed_positions, score_bucket_edges)
    exit_reasons = _build_exit_reasons(events_list)

    return AggregateReport(
        event_counts=event_counts,
        overall=overall,
        score_buckets=score_buckets,
        exit_reasons=exit_reasons,
        fees=fees,
        latency=latency,
        reject_reasons=dict(reject_reasons),
        paper_fail_buy=paper_fail_buy,
        paper_fail_sell=paper_fail_sell,
    )


def format_report_text(report: AggregateReport) -> str:
    """Render an :class:`AggregateReport` as a human-readable table.

    Plain ASCII, no Unicode box-drawing, so it works in any terminal /
    CI log / Discord paste. Each section is preceded by a one-line
    header so a reader can grep for it.
    """
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("tsuki-pump performance report")
    lines.append("=" * 72)

    # ── Event counts ────────────────────────────────────────────────
    lines.append("")
    lines.append(f"[events]  total: {report.event_counts.total}")
    if report.event_counts.by_kind:
        max_key = max(len(k) for k in report.event_counts.by_kind)
        for kind, n in sorted(report.event_counts.by_kind.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {kind.ljust(max_key)}  {n:>6}")
    else:
        lines.append("  (no events recorded)")

    # ── Failures + rejects ──────────────────────────────────────────
    lines.append("")
    lines.append(
        f"[reliability]  paper.buy_failed: {report.paper_fail_buy}   "
        f"paper.sell_failed: {report.paper_fail_sell}"
    )
    if report.reject_reasons:
        lines.append("  risk-engine reject reasons:")
        for reason, n in sorted(report.reject_reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {reason}: {n}")

    # ── Closed-position P&L ─────────────────────────────────────────
    lines.append("")
    lines.append("[closed positions — overall]")
    lines.extend(_render_bucket_stats(report.overall, indent="  "))

    # ── By score bucket ─────────────────────────────────────────────
    lines.append("")
    lines.append("[closed positions — by score_at_entry bucket]")
    if not any(b.stats.n_positions for b in report.score_buckets):
        lines.append("  (no closed positions to bucket)")
    else:
        header = f"  {'bucket':<14} {'n':>5} {'wins':>5} {'win%':>6} {'median':>9} {'mean':>9} {'total':>9}"
        lines.append(header)
        for b in report.score_buckets:
            s = b.stats
            if s.n_positions == 0:
                continue
            lines.append(
                f"  {b.label:<14} {s.n_positions:>5} {s.n_winners:>5} {s.win_rate * 100:>5.0f}%"
                f" {s.median_pnl_sol:>9.4f} {s.mean_pnl_sol:>9.4f} {s.total_pnl_sol:>9.4f}"
            )

    # ── Exit reasons ────────────────────────────────────────────────
    lines.append("")
    lines.append("[exits — by reason]")
    if not report.exit_reasons:
        lines.append("  (no paper.sell events recorded)")
    else:
        lines.append(f"  {'reason':<24} {'n':>5} {'total_pnl':>11} {'mean_pnl':>11}")
        for r in report.exit_reasons:
            lines.append(
                f"  {r.reason:<24} {r.n_exits:>5} {r.total_pnl_sol:>11.4f} {r.mean_pnl_sol:>11.4f}"
            )

    # ── Fees + drift ────────────────────────────────────────────────
    lines.append("")
    lines.append("[paper-realism friction]")
    if report.fees.fills_observed == 0:
        lines.append("  (no fills with diagnostic fields — enable execution.paper_realism)")
    else:
        lines.append(f"  fills observed:          {report.fees.fills_observed}")
        lines.append(f"  pump.fun fee total:      {report.fees.pump_fee_sol_total:.6f} SOL")
        lines.append(f"  priority fee total:      {report.fees.priority_fee_sol_total:.6f} SOL")
        lines.append(f"  drift cost total:        {report.fees.drift_sol_absorbed_total:.6f} SOL")
        lines.append(f"  ── total friction:       {report.fees.total_friction_sol:.6f} SOL")

    # ── Latency ─────────────────────────────────────────────────────
    if report.latency is not None and report.latency.samples > 0:
        lines.append("")
        lines.append("[fill latency]")
        lines.append(
            f"  samples: {report.latency.samples}   "
            f"p50: {report.latency.p50_ms:.0f} ms   "
            f"p90: {report.latency.p90_ms:.0f} ms   "
            f"p99: {report.latency.p99_ms:.0f} ms   "
            f"max: {report.latency.max_ms:.0f} ms"
        )

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


# ── helpers ──────────────────────────────────────────────────────────────


def _load_payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = event.get("payload_json") or event.get("payload")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        if isinstance(loaded, dict):
            return loaded
    return {}


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 0.0  # treat bools as 0 so a stray True doesn't pollute totals
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _summarise_latency(samples: list[float]) -> LatencyStats | None:
    if not samples:
        return None
    sorted_samples = sorted(samples)
    return LatencyStats(
        p50_ms=_percentile(sorted_samples, 0.50),
        p90_ms=_percentile(sorted_samples, 0.90),
        p99_ms=_percentile(sorted_samples, 0.99),
        max_ms=sorted_samples[-1],
        samples=len(sorted_samples),
    )


def _percentile(sorted_samples: list[float], q: float) -> float:
    if not sorted_samples:
        return 0.0
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    pos = q * (len(sorted_samples) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_samples[lo]
    weight = pos - lo
    return sorted_samples[lo] * (1.0 - weight) + sorted_samples[hi] * weight


def _bucket_stats(positions: list[Mapping[str, Any]]) -> BucketStats:
    if not positions:
        return BucketStats(
            n_positions=0,
            n_winners=0,
            win_rate=0.0,
            median_pnl_sol=0.0,
            mean_pnl_sol=0.0,
            total_pnl_sol=0.0,
            pnl_sharpe=None,
        )
    pnls = [_as_float(p.get("realized_pnl_sol")) for p in positions]
    n_winners = sum(1 for x in pnls if x > 0)
    total = sum(pnls)
    mean = total / len(pnls)
    median = _percentile(sorted(pnls), 0.50)
    sharpe: float | None = None
    if len(pnls) >= 2:
        variance = sum((x - mean) ** 2 for x in pnls) / (len(pnls) - 1)
        stdev = math.sqrt(variance)
        if stdev > 0:
            sharpe = mean / stdev
    return BucketStats(
        n_positions=len(pnls),
        n_winners=n_winners,
        win_rate=n_winners / len(pnls),
        median_pnl_sol=median,
        mean_pnl_sol=mean,
        total_pnl_sol=total,
        pnl_sharpe=sharpe,
    )


def _build_score_buckets(
    positions: list[Mapping[str, Any]], edges: tuple[float, ...]
) -> tuple[ScoreBucket, ...]:
    if len(edges) < 2:
        raise ValueError("score_bucket_edges must have at least two values")
    buckets: list[ScoreBucket] = []
    sorted_edges = sorted(edges)
    for lo, hi in pairwise(sorted_edges):
        in_bucket = [
            p
            for p in positions
            if p.get("score_at_entry") is not None and lo <= _as_float(p.get("score_at_entry")) < hi
        ]
        label = f"{lo:.0f}-{hi:.0f}" if hi < 100 else f"{lo:.0f}+"
        buckets.append(
            ScoreBucket(
                label=label,
                score_min=lo,
                score_max=hi,
                stats=_bucket_stats(in_bucket),
            )
        )
    return tuple(buckets)


def _build_exit_reasons(events: list[Mapping[str, Any]]) -> tuple[ExitReasonStats, ...]:
    """Aggregate paper.sell events by their ``reason`` payload field."""
    by_reason: dict[str, list[float]] = defaultdict(list)
    for ev in events:
        if str(ev.get("kind") or "") != "paper.sell":
            continue
        payload = _load_payload(ev)
        reason = str(payload.get("reason") or "unknown")
        pnl = _as_float(payload.get("pnl_sol"))
        by_reason[reason].append(pnl)
    out: list[ExitReasonStats] = []
    for reason, pnls in sorted(by_reason.items(), key=lambda kv: -sum(kv[1])):
        total = sum(pnls)
        out.append(
            ExitReasonStats(
                reason=reason,
                n_exits=len(pnls),
                total_pnl_sol=total,
                mean_pnl_sol=total / len(pnls) if pnls else 0.0,
            )
        )
    return tuple(out)


def _render_bucket_stats(stats: BucketStats, *, indent: str) -> list[str]:
    if stats.n_positions == 0:
        return [f"{indent}(no closed positions)"]
    sharpe_str = "n/a" if stats.pnl_sharpe is None else f"{stats.pnl_sharpe:.2f}"
    return [
        f"{indent}n: {stats.n_positions}   winners: {stats.n_winners}   "
        f"win-rate: {stats.win_rate * 100:.0f}%",
        f"{indent}total pnl: {stats.total_pnl_sol:+.4f} SOL   "
        f"mean: {stats.mean_pnl_sol:+.4f} SOL   "
        f"median: {stats.median_pnl_sol:+.4f} SOL",
        f"{indent}pnl-sharpe (per-trade mean/stdev): {sharpe_str}",
    ]
