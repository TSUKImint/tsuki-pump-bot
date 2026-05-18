"""Watchtower orchestrator — wires every component together.

Flow:
  1. PumpScout streams events from the firehose.
  2. TokenStateAggregator folds them into TokenState objects.
  3. A scoring loop iterates over tokens periodically and runs every filter.
  4. The composite scorer combines filter outcomes.
  5. If the score crosses the entry threshold AND we're in paper-trader mode,
     PaperExecutor simulates a buy and risk_engine records it.
  6. PositionMonitor checks exits each cycle and triggers paper sells.
  7. EventStore persists everything; DashboardState exposes a live view.
  8. Telegram broadcasts entries / exits / kill-switch trips.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

from .config import Config, Settings
from .core.event_store import EventStore
from .core.killswitch import KillSwitch
from .core.risk import RiskEngine, RiskLimits
from .core.sizing import SizingInputs, size_memecoin_position
from .core.telegram import TelegramClient
from .execution.paper_executor import PaperExecutor, PaperFill
from .execution.position_monitor import ExitAction, Position, PositionMonitor
from .filters import (
    BundleDetector,
    ConvergenceDetector,
    CtoDetector,
    CurvePredictor,
    DevBlacklist,
    FirstKolTouch,
)
from .models import FilterOutcome, TokenState
from .scoring import CompositeScorer
from .scout.aggregator import TokenStateAggregator
from .scout.pump_scout import PumpScout
from .solana.bonding_curve import (
    DEFAULT_VIRTUAL_SOL_RESERVES,
    DEFAULT_VIRTUAL_TOKEN_RESERVES,
    LAMPORTS_PER_SOL,
    BondingCurveState,
)

logger = structlog.get_logger(__name__)


@dataclass
class OrchestratorContext:
    """Bundle of every long-lived object the orchestrator owns.

    Constructed at startup, torn down on shutdown. Passed by reference into
    coroutines so tests can replace pieces wholesale.
    """

    settings: Settings
    config: Config
    kill: KillSwitch
    event_store: EventStore
    telegram: TelegramClient
    risk: RiskEngine
    aggregator: TokenStateAggregator
    composite_scorer: CompositeScorer
    paper_executor: PaperExecutor
    position_monitor: PositionMonitor
    dev_blacklist: DevBlacklist
    bundle_detector: BundleDetector
    first_kol_touch: FirstKolTouch
    convergence: ConvergenceDetector
    curve_predictor: CurvePredictor
    cto_detector: CtoDetector
    started_at: datetime
    open_positions: dict[str, Position]
    paper_trader_enabled: bool


def build_risk_engine(config: Config) -> RiskEngine:
    limits = RiskLimits(
        daily_drawdown_kill=config.bankroll.daily_drawdown_kill,
        total_drawdown_kill=config.bankroll.total_drawdown_kill,
        single_token_cap_fraction=config.bankroll.single_token_cap_fraction,
        max_open_positions=config.bankroll.max_open_positions,
        hard_cap_per_trade_sol=config.sizing.hard_cap_per_trade_sol,
    )
    return RiskEngine(starting_bankroll_sol=config.bankroll.total_sol, limits=limits)


async def run_scoring_loop(
    ctx: OrchestratorContext,
    dashboard_publish: Callable[[list[TokenState]], None] | None = None,
    *,
    cycle_seconds: float = 5.0,
) -> None:
    """Run filter+score over the aggregator's token set every `cycle_seconds`.

    Returns when the kill switch trips.
    """
    while not ctx.kill.tripped:
        try:
            await _score_once(ctx)
            if dashboard_publish is not None:
                top = sorted(
                    ctx.aggregator.all_tokens(),
                    key=lambda t: t.last_composite_score,
                    reverse=True,
                )[:20]
                dashboard_publish(top)
        except Exception as exc:
            logger.error("orchestrator.score_loop_error", err=str(exc))

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(ctx.kill.wait_for_trip(), timeout=cycle_seconds)


async def run_position_loop(
    ctx: OrchestratorContext,
    *,
    cycle_seconds: float = 4.0,
) -> None:
    """Evaluate exits across open paper positions every `cycle_seconds`."""
    while not ctx.kill.tripped:
        try:
            await _check_exits_once(ctx)
        except Exception as exc:
            logger.error("orchestrator.position_loop_error", err=str(exc))

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(ctx.kill.wait_for_trip(), timeout=cycle_seconds)


async def run_scout_loop(
    ctx: OrchestratorContext,
    scout: PumpScout,
) -> None:
    """Drain the firehose and feed events into the aggregator."""
    async for event in scout.stream():
        if ctx.kill.tripped:
            break
        try:
            ctx.aggregator.ingest(event)
        except Exception as exc:
            logger.warning("orchestrator.ingest_error", err=str(exc))


# ── inner helpers ──────────────────────────────────────────────────────────


async def _score_once(ctx: OrchestratorContext) -> None:
    """Run filters + composite scoring across all known tokens."""
    for token in ctx.aggregator.all_tokens():
        outcomes = await _run_filters(ctx, token)
        composite = ctx.composite_scorer.score(outcomes)
        token.last_composite_score = composite.score
        token.last_filter_scores = composite.breakdown
        token.rejected = composite.hard_rejected
        token.reject_reason = composite.reject_reason
        if composite.hard_rejected:
            token.last_decision_reason = composite.reject_reason
        else:
            top_filter = max(outcomes, key=lambda o: o.score, default=None)
            token.last_decision_reason = (
                f"{top_filter.name}: {top_filter.notes}" if top_filter else ""
            )

        await ctx.event_store.upsert_token(
            mint=token.mint,
            dev_wallet=token.dev_wallet,
            symbol=token.symbol,
            name=token.name,
            score=composite.score,
            payload={
                "breakdown": composite.breakdown,
                "rejected": composite.hard_rejected,
                "reject_reason": composite.reject_reason,
            },
        )

        if not ctx.paper_trader_enabled:
            continue
        if composite.hard_rejected:
            continue
        if not composite.enter:
            continue
        if token.mint in ctx.open_positions:
            continue
        await _try_open_position(ctx, token, composite_score=composite.score)


async def _run_filters(
    ctx: OrchestratorContext,
    token: TokenState,
) -> list[FilterOutcome]:
    """Run every filter on a single token and return their outcomes."""
    outcomes: list[FilterOutcome] = []

    dev_token_count_24h, dev_token_count_7d = await _dev_token_counts(ctx, token.dev_wallet)
    outcomes.append(
        ctx.dev_blacklist.evaluate(
            token,
            dev_token_count_24h=dev_token_count_24h,
            dev_token_count_7d=dev_token_count_7d,
        )
    )
    outcomes.append(ctx.bundle_detector.evaluate(token, wallet_funder_lookup=None))
    outcomes.append(ctx.first_kol_touch.evaluate(token))
    outcomes.append(ctx.convergence.evaluate(token))
    outcomes.append(ctx.curve_predictor.evaluate(token))
    outcomes.append(
        ctx.cto_detector.evaluate(
            token,
            now_unix=int(datetime.now(tz=UTC).timestamp()),
            unique_buyers_24h=token.distinct_buyers_60s,  # crude proxy in v0.2
        )
    )
    return outcomes


async def _dev_token_counts(
    ctx: OrchestratorContext,
    dev_wallet: str | None,
) -> tuple[int, int]:
    """Count tokens launched by `dev_wallet` in the last 24h / 7d.

    Uses the local `tokens` table. Returns (24h_count, 7d_count).
    """
    if not dev_wallet:
        return 0, 0
    cursor = await ctx.event_store.db.execute(
        """
        SELECT first_seen_utc FROM tokens WHERE dev_wallet = ?
        """,
        (dev_wallet,),
    )
    rows = await cursor.fetchall()
    now = datetime.now(tz=UTC)
    count_24h = 0
    count_7d = 0
    for (ts,) in rows:
        try:
            seen = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            continue
        if seen.tzinfo is None:
            seen = seen.replace(tzinfo=UTC)
        age_seconds = (now - seen).total_seconds()
        if age_seconds <= 24 * 3600:
            count_24h += 1
        if age_seconds <= 7 * 86400:
            count_7d += 1
    return count_24h, count_7d


async def _try_open_position(
    ctx: OrchestratorContext,
    token: TokenState,
    *,
    composite_score: float,
) -> None:
    """Size + paper-fill a buy for `token`. No-op if any gate refuses."""
    cost_per_unit = max(1e-12, token.last_price_sol_per_token)
    inputs = SizingInputs(
        bankroll_sol=ctx.config.bankroll.total_sol,
        composite_score=composite_score,
        cost_per_unit_sol=cost_per_unit,
        fraction_of_kelly=ctx.config.sizing.fraction_of_kelly,
        hard_cap_per_trade_sol=ctx.config.sizing.hard_cap_per_trade_sol,
        single_token_cap_sol=(
            ctx.config.bankroll.total_sol * ctx.config.bankroll.single_token_cap_fraction
        ),
    )
    sizing = size_memecoin_position(inputs)
    if sizing.units <= 0 or sizing.notional_sol <= 0:
        return

    decision = ctx.risk.check_open_allowed(sizing.notional_sol)
    if not decision.allowed:
        await ctx.event_store.record_event(
            kind="risk.reject",
            summary=f"refused buy for {token.mint[:8]}: {decision.reason}",
            mint=token.mint,
            severity="warning",
        )
        return

    curve = _synthesize_curve_from_token(token)
    inflow_sol_per_sec = max(0.0, token.last_sol_velocity_sol_per_min) / 60.0
    try:
        fill = ctx.paper_executor.buy(
            token.mint,
            sizing.units,
            curve,
            observed_inflow_sol_per_sec=inflow_sol_per_sec,
        )
    except ValueError as exc:
        await ctx.event_store.record_event(
            kind="paper.buy_failed",
            summary=f"paper buy failed for {token.mint[:8]}: {exc}",
            mint=token.mint,
            severity="warning",
        )
        return

    ctx.risk.record_open(fill.notional_sol)
    position_id = await ctx.event_store.open_position(
        mint=token.mint,
        dev_wallet=token.dev_wallet,
        entry_units=fill.fill_units,
        entry_price_sol=fill.fill_price_sol_per_token,
        entry_notional_sol=fill.notional_sol,
        paper=True,
        score_at_entry=composite_score,
        payload={
            "score_breakdown": token.last_filter_scores,
            "sizing": {
                "kelly_raw": sizing.kelly_fraction_raw,
                "binding": sizing.binding_constraint,
                "units": sizing.units,
            },
        },
    )
    ctx.open_positions[token.mint] = Position(
        mint=token.mint,
        units_held=fill.fill_units,
        entry_units=fill.fill_units,
        entry_price_sol_per_token=fill.fill_price_sol_per_token,
        peak_price_sol_per_token=fill.fill_price_sol_per_token,
    )
    await ctx.event_store.record_event(
        kind="paper.buy",
        summary=(
            f"paper BUY {fill.fill_units:.4f} of {token.mint[:8]} @ "
            f"{fill.fill_price_sol_per_token:.8f} SOL (notional {fill.notional_sol:.4f} SOL, "
            f"score {composite_score:.1f})"
        ),
        mint=token.mint,
        dev_wallet=token.dev_wallet,
        payload={"position_id": position_id, "fill": fill.__dict__},
    )
    if ctx.config.telegram.notify_on_entry:
        await ctx.telegram.send_message(
            f"🟢 PAPER ENTRY\n"
            f"mint: {token.mint}\n"
            f"score: {composite_score:.1f}\n"
            f"units: {fill.fill_units:.4f}\n"
            f"price: {fill.fill_price_sol_per_token:.8f} SOL\n"
            f"notional: {fill.notional_sol:.4f} SOL"
        )


async def _check_exits_once(ctx: OrchestratorContext) -> None:
    """Iterate open positions and trigger exits per the configured policy."""
    if not ctx.open_positions:
        return
    for mint, pos in list(ctx.open_positions.items()):
        token = ctx.aggregator.get(mint)
        if token is None:
            continue
        current_price = token.last_price_sol_per_token
        if current_price <= 0:
            continue
        dev_drained_fraction = _dev_drained_fraction(token)
        actions = ctx.position_monitor.evaluate(
            pos, current_price, dev_drained_fraction=dev_drained_fraction
        )
        if not actions:
            continue
        for action in actions:
            await _execute_exit(ctx, token, pos, action, current_price)
        if pos.units_held <= 1e-12:
            await _close_position_row(ctx, pos)


def _dev_drained_fraction(token: TokenState) -> float:
    """Approximate fraction of dev's initial SOL drained out of the curve.

    In paper-mock mode we don't have a precise initial-balance reading, so
    we treat any cumulative SOL pulled as the drained amount and compare to
    a heuristic floor (10 SOL = "meaningful drain"). This favours false
    negatives over false positives; live mode will read the actual balance.
    """
    pulled = token.dev_activity.cumulative_sol_pulled
    if pulled <= 0:
        return 0.0
    initial = token.dev_activity.sol_balance_at_launch or 10.0
    return min(1.0, pulled / max(1.0, initial))


async def _execute_exit(
    ctx: OrchestratorContext,
    token: TokenState,
    pos: Position,
    action: ExitAction,
    current_price: float,
) -> None:
    """Run a paper sell for `action.units_to_sell`."""
    curve = _synthesize_curve_from_token(token)
    inflow_sol_per_sec = max(0.0, token.last_sol_velocity_sol_per_min) / 60.0
    try:
        fill: PaperFill = ctx.paper_executor.sell(
            token.mint,
            action.units_to_sell,
            curve,
            observed_inflow_sol_per_sec=inflow_sol_per_sec,
        )
    except ValueError as exc:
        await ctx.event_store.record_event(
            kind="paper.sell_failed",
            summary=f"paper sell failed for {token.mint[:8]}: {exc}",
            mint=token.mint,
            severity="warning",
        )
        return
    cost_basis_per_unit = pos.entry_price_sol_per_token
    pnl = (fill.fill_price_sol_per_token - cost_basis_per_unit) * fill.fill_units
    pos.units_held = max(0.0, pos.units_held - fill.fill_units)
    pos.realized_pnl_sol += pnl
    ctx.risk.record_close(pnl, notional_freed_sol=cost_basis_per_unit * fill.fill_units)
    await ctx.event_store.record_event(
        kind="paper.sell",
        summary=(
            f"paper SELL {fill.fill_units:.4f} of {token.mint[:8]} @ "
            f"{fill.fill_price_sol_per_token:.8f} SOL "
            f"(pnl {pnl:+.4f} SOL, reason {action.reason})"
        ),
        mint=token.mint,
        dev_wallet=token.dev_wallet,
        payload={
            "fill": fill.__dict__,
            "pnl_sol": pnl,
            "reason": action.reason,
            "remaining_units": pos.units_held,
            "current_price": current_price,
        },
    )
    if ctx.config.telegram.notify_on_exit:
        await ctx.telegram.send_message(
            f"🔴 PAPER EXIT [{action.reason}]\n"
            f"mint: {token.mint}\n"
            f"sold: {fill.fill_units:.4f} units\n"
            f"price: {fill.fill_price_sol_per_token:.8f} SOL\n"
            f"pnl: {pnl:+.4f} SOL"
        )


async def _close_position_row(ctx: OrchestratorContext, pos: Position) -> None:
    cursor = await ctx.event_store.db.execute(
        "SELECT id FROM positions WHERE mint=? AND status='open'",
        (pos.mint,),
    )
    row = await cursor.fetchone()
    if row:
        await ctx.event_store.close_position(int(row[0]), realized_pnl_sol=pos.realized_pnl_sol)
    ctx.open_positions.pop(pos.mint, None)


def _synthesize_curve_from_token(token: TokenState) -> BondingCurveState:
    """Build a BondingCurveState approximation from the aggregator's view.

    Live mode will replace this with a real BondingCurve account read.
    """
    real_sol_lamports = int(token.sol_in_curve * LAMPORTS_PER_SOL)
    return BondingCurveState(
        virtual_sol_reserves=DEFAULT_VIRTUAL_SOL_RESERVES + real_sol_lamports,
        virtual_token_reserves=DEFAULT_VIRTUAL_TOKEN_RESERVES,
        real_sol_reserves=real_sol_lamports,
        real_token_reserves=0,
        complete=token.curve_complete,
    )
