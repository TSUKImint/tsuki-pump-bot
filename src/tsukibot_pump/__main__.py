"""Bot entrypoint — argparse + orchestrator wiring + Windows-safe signals.

$ tsuki-pump                     # paper mode + dashboard (default)
$ tsuki-pump --mode paper-mock   # no RPC at all (firehose disabled)
$ tsuki-pump --no-dashboard      # plain logs, friendly to nohup / systemd
$ tsuki-pump --once              # one scoring cycle, then exit (for tests)
$ tsuki-pump --i-understand-this-trades-real-money   # gate for devnet/mainnet
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import structlog

from . import __version__
from .cli.dashboard import Dashboard, DashboardState
from .config import Config, Settings, load_config
from .core.event_store import EventStore
from .core.killswitch import KillSwitch
from .core.telegram import TelegramClient
from .execution.paper_executor import PaperExecutor
from .execution.position_monitor import Position, PositionMonitor
from .filters import (
    BundleDetector,
    ConvergenceDetector,
    CtoDetector,
    CurvePredictor,
    DevBlacklist,
    FirstKolTouch,
)
from .firehose import FirehoseRecorder, FirehoseReplayer, ReplaySpeedMode
from .logging import configure_logging, get_logger
from .models import TokenState
from .orchestrator import (
    OrchestratorContext,
    build_risk_engine,
    run_position_loop,
    run_scoring_loop,
    run_scout_loop,
)
from .scoring import CompositeScorer
from .scout.aggregator import TokenStateAggregator
from .scout.pump_scout import PumpScout
from .solana.rpc import SolanaRPCClient


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tsuki-pump",
        description=(
            "Pump.fun watchtower + paper-trader. "
            "Selection-edge bot. Paper-default. See PUMPFUN_PLAN.md."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--mode",
        choices=["paper-mock", "paper", "devnet", "mainnet"],
        default=None,
        help="Override TSUKI_PUMP_MODE.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config (overrides TSUKI_PUMP_CONFIG).",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Run without the live TUI (good for `nohup` / systemd / Docker).",
    )
    parser.add_argument(
        "--paper-trader",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable paper buys/sells when score crosses the threshold. "
        "Default ON. Pass --no-paper-trader to run watchtower-only.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one scoring cycle and exit (for smoke tests).",
    )
    parser.add_argument(
        "--i-understand-this-trades-real-money",
        action="store_true",
        help="Required for devnet / mainnet modes (alongside an explicit --mode).",
    )
    parser.add_argument(
        "--replay-firehose",
        type=Path,
        default=None,
        help=(
            "Backtest mode: replay a recorded JSONL firehose instead of "
            "connecting to RPC. Disables the dashboard and any live exec."
        ),
    )
    parser.add_argument(
        "--replay-speed",
        choices=[m.value for m in ReplaySpeedMode],
        default=ReplaySpeedMode.ASAP.value,
        help="Replay speed mode (default: asap — deterministic, no sleeps).",
    )
    parser.add_argument(
        "--replay-multiplier",
        type=float,
        default=60.0,
        help="Speed multiplier for 'compressed' replay mode (default 60x).",
    )
    parser.add_argument(
        "--record-firehose",
        type=Path,
        default=None,
        help="Live mode: also write every observed event to this JSONL path.",
    )
    return parser.parse_args(argv)


# ── Signal handling ────────────────────────────────────────────────────────


def _make_signal_handler(sig: signal.Signals, kill: KillSwitch) -> Callable[[], None]:
    def _handler() -> None:
        kill.trip(f"signal {sig.name}")

    return _handler


def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    kill: KillSwitch,
    logger: structlog.stdlib.BoundLogger,
) -> None:
    """Wire SIGINT/SIGTERM to the kill switch where the platform supports it.

    On Windows, `add_signal_handler` raises `NotImplementedError`. In that
    case we silently fall back to the default behaviour: `asyncio.run()`
    turns `Ctrl+C` into a `KeyboardInterrupt`, which `main()` catches and
    converts to exit code 130. The `finally` block in `_async_main` trips
    the kill switch unconditionally so cleanup still runs.
    """
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _make_signal_handler(sig, kill))
        except NotImplementedError:
            logger.debug(
                "signal.handler.unsupported_on_platform",
                signal=sig.name,
                note="ctrl+c still works via KeyboardInterrupt path",
            )


# ── Gates ──────────────────────────────────────────────────────────────────


def _check_live_gate(
    settings: Settings,
    args: argparse.Namespace,
    logger: structlog.stdlib.BoundLogger,
) -> int:
    """Refuse to start in devnet/mainnet without the explicit ack flag."""
    if not settings.places_real_orders:
        return 0
    if not args.i_understand_this_trades_real_money:
        logger.error(
            "live_gate.refuse",
            mode=settings.tsuki_pump_mode,
            note=(
                "Devnet/mainnet modes require --i-understand-this-trades-real-money. "
                "This is intentionally tedious — re-read PUMPFUN_PLAN.md "
                "before you bypass paper mode."
            ),
        )
        return 2
    if settings.risks_real_money and not settings.solana_hot_wallet_secret:
        logger.error(
            "live_gate.no_wallet",
            note="Mainnet mode requires SOLANA_HOT_WALLET_SECRET in .env",
        )
        return 2
    return 0


# ── Main flow ──────────────────────────────────────────────────────────────


async def _async_main(args: argparse.Namespace) -> int:
    settings = Settings()
    if args.mode:
        settings = settings.model_copy(update={"tsuki_pump_mode": args.mode})
    if args.config:
        settings = settings.model_copy(update={"tsuki_pump_config": args.config})

    configure_logging(settings.tsuki_pump_log_level, state_dir=settings.tsuki_pump_state_dir)
    logger = get_logger("tsuki-pump")
    logger.info(
        "tsuki-pump.start",
        version=__version__,
        mode=settings.tsuki_pump_mode,
        config_path=str(settings.tsuki_pump_config),
        dashboard=not args.no_dashboard,
        paper_trader=args.paper_trader,
    )

    gate_exit = _check_live_gate(settings, args, logger)
    if gate_exit != 0:
        return gate_exit

    try:
        config = load_config(settings.tsuki_pump_config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("config.load_failed", err=str(exc))
        return 2

    kill = KillSwitch()
    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop, kill, logger)

    try:
        async with (
            EventStore(
                config.event_store.sqlite_path,
                config.event_store.jsonl_audit_path,
            ) as event_store,
            TelegramClient(
                settings.telegram_bot_token,
                settings.telegram_chat_id,
                http_timeout_seconds=config.network.http_timeout_seconds,
            ) as telegram,
        ):
            return await _run(args, settings, config, logger, kill, event_store, telegram)
    except KeyboardInterrupt:
        logger.info("tsuki-pump.keyboard_interrupt")
        return 130
    finally:
        # Trip kill switch unconditionally — guarantees that nested tasks
        # see shutdown even on exceptions that skipped the signal handlers.
        kill.trip("orchestrator exit")


async def _run(
    args: argparse.Namespace,
    settings: Settings,
    config: Config,
    logger: structlog.stdlib.BoundLogger,
    kill: KillSwitch,
    event_store: EventStore,
    telegram: TelegramClient,
) -> int:
    risk = build_risk_engine(config)
    aggregator = TokenStateAggregator()
    composite_scorer = CompositeScorer(config.scoring)
    paper_executor = PaperExecutor(
        slippage_bps=config.execution.paper_slippage_bps,
        realism=config.execution.paper_realism,
    )
    position_monitor = PositionMonitor(config.exits)

    dev_blacklist = DevBlacklist(config.filters.dev_blacklist)
    bundle_detector = BundleDetector(config.filters.bundle_cluster)
    first_kol_touch = FirstKolTouch(config.filters.first_kol_touch)
    convergence = ConvergenceDetector.from_kol_filter_config(
        config.filters.convergence, config.filters.first_kol_touch
    )
    curve_predictor = CurvePredictor(config.filters.curve_graduation)
    cto_detector = CtoDetector(config.filters.cto_revival)

    open_positions: dict[str, Position] = {}

    ctx = OrchestratorContext(
        settings=settings,
        config=config,
        kill=kill,
        event_store=event_store,
        telegram=telegram,
        risk=risk,
        aggregator=aggregator,
        composite_scorer=composite_scorer,
        paper_executor=paper_executor,
        position_monitor=position_monitor,
        dev_blacklist=dev_blacklist,
        bundle_detector=bundle_detector,
        first_kol_touch=first_kol_touch,
        convergence=convergence,
        curve_predictor=curve_predictor,
        cto_detector=cto_detector,
        started_at=datetime.now(tz=UTC),
        open_positions=open_positions,
        paper_trader_enabled=args.paper_trader
        and settings.tsuki_pump_mode
        in {
            "paper-mock",
            "paper",
        },
    )

    await event_store.record_event(
        kind="bot.start",
        summary=(
            f"tsuki-pump v{__version__} starting in {settings.tsuki_pump_mode} mode "
            f"(paper_trader={'on' if ctx.paper_trader_enabled else 'off'})"
        ),
        severity="info",
        payload={"version": __version__, "mode": settings.tsuki_pump_mode},
    )

    dash_state = DashboardState()
    dashboard: Dashboard | None = None
    if not args.no_dashboard:
        dashboard = Dashboard(
            mode=settings.tsuki_pump_mode,
            bankroll_sol=config.bankroll.total_sol,
            kill=kill,
            risk=risk,
            state=dash_state,
            refresh_hz=config.dashboard.refresh_hz,
            get_started_at=lambda: ctx.started_at,
        )

    # Wire telegram commands.
    async def _cmd_killswitch(_rest: str) -> None:
        kill.trip("telegram /killswitch")
        await telegram.send_message("kill switch tripped via telegram")

    async def _cmd_status(_rest: str) -> None:
        await telegram.send_message(
            f"<b>tsuki-pump status</b>\n"
            f"mode: {settings.tsuki_pump_mode}\n"
            f"open positions: {len(open_positions)}\n"
            f"realized pnl: {risk.pnl.realized_sol:+.4f} SOL\n"
            f"scout signatures seen: {dash_state.scout_stats.signatures_seen}"
        )

    telegram.register_command("killswitch", _cmd_killswitch)
    telegram.register_command("status", _cmd_status)
    telegram.start_polling()
    if telegram.configured:
        await telegram.send_message(
            f"🟢 tsuki-pump v{__version__} starting ({settings.tsuki_pump_mode})"
        )

    def _publish(top: list[TokenState]) -> None:
        dash_state.top_tokens = top
        dash_state.open_positions = dict(ctx.open_positions)
        dash_state.current_prices = {
            t.mint: t.last_price_sol_per_token for t in aggregator.all_tokens()
        }
        dash_state.tokens_scored_total = len(aggregator.all_tokens())
        dash_state.tokens_rejected_total = sum(1 for t in aggregator.all_tokens() if t.rejected)

    # ── --once: deterministic one-shot for smoke tests ────────────────────
    if args.once:
        from .orchestrator import _score_once

        await _score_once(ctx)
        _publish(
            sorted(
                aggregator.all_tokens(),
                key=lambda t: t.last_composite_score,
                reverse=True,
            )[:20]
        )
        logger.info("tsuki-pump.once_complete", tokens=len(aggregator))
        return 0

    # ── Run the live system ──────────────────────────────────────────────
    tasks: list[asyncio.Task[object]] = []

    scoring_task = asyncio.create_task(
        run_scoring_loop(ctx, _publish, cycle_seconds=5.0),
        name="scoring",
    )
    tasks.append(scoring_task)

    position_task = asyncio.create_task(
        run_position_loop(ctx, cycle_seconds=4.0),
        name="positions",
    )
    tasks.append(position_task)

    if args.replay_firehose is not None:
        # Backtest mode: yield events from a recorded JSONL instead of RPC.
        # No RPC connection, no Telegram, no live exec — just the same
        # scoring + paper-trading pipeline driven by a deterministic feed.
        if not args.replay_firehose.exists():
            logger.error("tsuki-pump.replay_file_missing", path=str(args.replay_firehose))
            return 2
        replay_stop = asyncio.Event()
        replayer = FirehoseReplayer(
            args.replay_firehose,
            mode=args.replay_speed,
            speed_multiplier=args.replay_multiplier,
            stop_event=replay_stop,
        )
        scout_task = asyncio.create_task(run_scout_loop(ctx, replayer), name="replay")
        tasks.append(scout_task)
        if dashboard is not None:
            dash_task = asyncio.create_task(dashboard.run_forever(), name="dashboard")
            tasks.append(dash_task)
        try:
            # Stop when either the replay is exhausted or the kill switch
            # trips. We don't want one to keep the bot alive after the
            # other has decided to shut down.
            done, _pending = await asyncio.wait(
                {scout_task, asyncio.create_task(kill.wait_for_trip(), name="kill_wait")},
                return_when=asyncio.FIRST_COMPLETED,
            )
            _ = done
        finally:
            replay_stop.set()
    elif settings.is_live_chain:
        rpc = SolanaRPCClient(
            settings.effective_rpc_url,
            http_timeout_seconds=config.network.http_timeout_seconds,
            requests_per_second=config.network.rpc_requests_per_second,
        )
        await rpc.__aenter__()  # entered manually so we can cancel cleanly
        try:
            base_scout = PumpScout(
                rpc,
                poll_interval_seconds=config.watch.http_poll_interval_seconds,
                stop_event=asyncio.Event(),
            )
            # Pump scout stats publish into dashboard via shared reference.
            dash_state.scout_stats = base_scout.stats

            scout_source: PumpScout | FirehoseRecorder
            if args.record_firehose is not None:
                scout_source = FirehoseRecorder(base_scout, output_path=args.record_firehose)
                logger.info(
                    "tsuki-pump.record_firehose_enabled",
                    out=str(args.record_firehose),
                )
            else:
                scout_source = base_scout

            scout_task = asyncio.create_task(run_scout_loop(ctx, scout_source), name="scout")
            tasks.append(scout_task)

            if dashboard is not None:
                dash_task = asyncio.create_task(dashboard.run_forever(), name="dashboard")
                tasks.append(dash_task)

            await kill.wait_for_trip()
        finally:
            await rpc.__aexit__(None, None, None)
    else:
        # paper-mock: no RPC, no firehose. The scoring loop will run on an
        # initially-empty aggregator; useful for verifying plumbing without
        # a network connection.
        if dashboard is not None:
            dash_task = asyncio.create_task(dashboard.run_forever(), name="dashboard")
            tasks.append(dash_task)
        await kill.wait_for_trip()

    for t in tasks:
        t.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.gather(*tasks, return_exceptions=True)

    await event_store.record_event(
        kind="bot.stop",
        summary=f"tsuki-pump stopped: {kill.reason}",
        severity="info",
        payload={"reason": kill.reason},
    )
    if telegram.configured and config.telegram.notify_on_kill_switch:
        await telegram.send_message(f"⏹ tsuki-pump stopped: {kill.reason}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
