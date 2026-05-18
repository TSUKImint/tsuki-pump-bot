"""CLI: ``python -m tsukibot_pump.firehose record|replay``.

Two subcommands:

* ``record`` — connect to pump.fun via :class:`PumpScout` and write
  every event to a JSONL file until interrupted. The file becomes a
  reusable input for ``--replay-firehose`` in the main bot CLI and
  for ``python -m tsukibot_pump.discovery``.
* ``replay`` — read a JSONL recording and print each event (for
  smoke-testing a recording without launching the full bot).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
from pathlib import Path

import structlog

from ..config import Settings, load_config
from ..logging import configure_logging
from ..scout.pump_scout import PumpScout
from ..solana.rpc import SolanaRPCClient
from .recorder import FirehoseRecorder
from .replayer import FirehoseReplayer, ReplaySpeedMode

logger = structlog.get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tsukibot_pump.firehose",
        description="Record or replay a pump.fun event firehose for backtesting.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="Record events from PumpScout to JSONL.")
    rec.add_argument("--out", required=True, type=Path, help="Output JSONL path (append mode).")
    rec.add_argument(
        "--config",
        default=None,
        help="Path to YAML config (defaults to TSUKI_PUMP_CONFIG / config.yaml).",
    )
    rec.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Stop after this many events (0 = unlimited, ^C to stop).",
    )

    rep = sub.add_parser("replay", help="Print events from a JSONL recording.")
    rep.add_argument(
        "--events", required=True, type=Path, help="Path to a JSONL firehose recording."
    )
    rep.add_argument(
        "--speed",
        choices=[m.value for m in ReplaySpeedMode],
        default=ReplaySpeedMode.ASAP.value,
    )
    rep.add_argument(
        "--multiplier",
        type=float,
        default=60.0,
        help="Speed multiplier for 'compressed' mode (default 60x).",
    )
    rep.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Stop after this many events (0 = play to end).",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args(argv)


async def _run_record(args: argparse.Namespace) -> int:
    config = load_config(path=args.config)
    settings = Settings()

    rpc = SolanaRPCClient(
        settings.effective_rpc_url,
        http_timeout_seconds=config.network.http_timeout_seconds,
        requests_per_second=config.network.rpc_requests_per_second,
    )
    stop_event = asyncio.Event()

    def _on_signal() -> None:
        logger.info("firehose.record.shutdown_requested")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, _on_signal)

    async with rpc:
        scout = PumpScout(
            rpc,
            poll_interval_seconds=config.watch.http_poll_interval_seconds,
            stop_event=stop_event,
        )
        recorder = FirehoseRecorder(scout, output_path=args.out)
        logger.info(
            "firehose.record.started",
            out=str(args.out),
            poll_interval=config.watch.http_poll_interval_seconds,
        )
        try:
            async for _event in recorder.stream():
                if args.max_events and recorder.events_recorded >= args.max_events:
                    stop_event.set()
                    break
        finally:
            recorder.close()
    logger.info("firehose.record.complete", events=recorder.events_recorded)
    return 0


async def _run_replay(args: argparse.Namespace) -> int:
    if not args.events.exists():
        logger.error("firehose.replay.file_missing", path=str(args.events))
        return 2

    replayer = FirehoseReplayer(
        args.events,
        mode=args.speed,
        speed_multiplier=args.multiplier,
    )
    async for event in replayer.stream():
        # Plain-text dump so users can `python -m … replay | head`.
        print(
            f"{event.kind.value:<6} slot={event.slot} "
            f"mint={event.mint[:8] if event.mint else '----'} "
            f"actor={event.actor_wallet[:8] if event.actor_wallet else '----'} "
            f"sol={event.sol_amount or 0:.4f}"
        )
        if args.max_events and replayer.stats.events_emitted >= args.max_events:
            break
    logger.info("firehose.replay.complete", events=replayer.stats.events_emitted)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging(log_level=args.log_level, json_console=False)

    if args.command == "record":
        return asyncio.run(_run_record(args))
    if args.command == "replay":
        return asyncio.run(_run_replay(args))
    return 1


if __name__ == "__main__":
    sys.exit(main())
