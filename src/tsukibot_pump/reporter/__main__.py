"""CLI entry point: ``python -m tsukibot_pump.reporter``.

Reads the bot's SQLite event store and prints a per-cohort
performance report. Run after a paper-trading session to see whether
the strategy actually has edge and where the spread is going.

    # Default DB path (state/tsuki_pump.sqlite).
    python -m tsukibot_pump.reporter

    # Custom DB path + JSON output.
    python -m tsukibot_pump.reporter --db state/audit.sqlite --json
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import structlog

from ..core.event_store import EventStore
from ..logging import configure_logging
from .report import AggregateReport, build_report, format_report_text

logger = structlog.get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tsukibot_pump.reporter",
        description="Summarise the bot's paper-trading event store.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("state/tsuki_pump.sqlite"),
        help="Path to the SQLite event store (default: state/tsuki_pump.sqlite).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of a human-readable table.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=100_000,
        help=(
            "Cap how many event rows to scan (default: 100k). "
            "Reduce on very large stores to keep the report fast."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
    )
    return parser.parse_args(argv)


async def _load_rows(
    db_path: Path, max_events: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit_jsonl = db_path.with_name(db_path.stem + "_audit.jsonl")
    async with EventStore(db_path, audit_jsonl) as store:
        events_cursor = await store.db.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (max_events,)
        )
        events_rows = await events_cursor.fetchall()
        events_cols = [d[0] for d in events_cursor.description]
        events = [dict(zip(events_cols, r, strict=True)) for r in events_rows]

        positions_cursor = await store.db.execute("SELECT * FROM positions ORDER BY id")
        positions_rows = await positions_cursor.fetchall()
        positions_cols = [d[0] for d in positions_cursor.description]
        positions = [dict(zip(positions_cols, r, strict=True)) for r in positions_rows]
    return events, positions


def _report_to_dict(report: AggregateReport) -> dict[str, Any]:
    return dataclasses.asdict(report)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging(log_level=args.log_level, json_console=False)

    if not args.db.exists():
        logger.error("reporter.db_missing", path=str(args.db))
        return 2

    events, positions = asyncio.run(_load_rows(args.db, args.max_events))
    report = build_report(events=events, positions=positions)
    if args.json:
        print(json.dumps(_report_to_dict(report), indent=2, sort_keys=True))
    else:
        print(format_report_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
