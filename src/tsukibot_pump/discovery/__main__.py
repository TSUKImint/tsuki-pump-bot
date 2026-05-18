"""CLI entry point: ``python -m tsukibot_pump.discovery``.

Reads pump.fun events from a JSONL firehose recording (the format
written by ``record_firehose.py`` — see PR #D in the improvement plan)
and emits a scored KOL CSV ready to be dropped at
``data/private_kol_list.csv``.

Two usage patterns:

  # Replay a recorded firehose into a fresh CSV.
  python -m tsukibot_pump.discovery \\
      --events state/firehose.jsonl \\
      --out data/private_kol_list.csv

  # Same, but supply funder / sell-destination lookups gathered out-of-band
  # to enable poison-wallet detection.
  python -m tsukibot_pump.discovery \\
      --events state/firehose.jsonl \\
      --funders state/funders.csv \\
      --sell-destinations state/sell_destinations.csv \\
      --out data/private_kol_list.csv

The funder CSV has the shape ``wallet,funder_wallet`` (one per line).
The sell-destination CSV has the shape ``wallet,destination_wallet``
with duplicate ``wallet`` rows accumulating the destination list.

Both are optional: when missing, poison detection runs in
"shared-funder-only" mode against any funder data inferred from the
events file itself (which is usually empty — PR #A keeps these
heuristics off by default until the user has supplied real data).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import structlog

from ..logging import configure_logging
from ..solana.pump_program import PumpEvent, PumpInstructionKind
from .kol_discovery import KolDiscoveryConfig, discover_kols, write_kol_csv

logger = structlog.get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="tsukibot_pump.discovery",
        description="Score wallets from a recorded pump.fun firehose.",
    )
    p.add_argument(
        "--events",
        required=True,
        type=Path,
        help="Path to a JSONL firehose recording (one PumpEvent dict per line).",
    )
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output CSV path (will overwrite). Format: wallet,label,score.",
    )
    p.add_argument(
        "--funders",
        type=Path,
        default=None,
        help="Optional CSV: wallet,funder_wallet — enables funder-graph poison check.",
    )
    p.add_argument(
        "--sell-destinations",
        type=Path,
        default=None,
        help="Optional CSV: wallet,destination_wallet (one row per destination).",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=200,
        help="Cap the leaderboard to this many wallets (default: 200).",
    )
    p.add_argument(
        "--now-unix",
        type=int,
        default=None,
        help=(
            "Unix timestamp to treat as 'now' for the recency cut. "
            "Defaults to current wall-clock time."
        ),
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return p.parse_args(argv)


def _read_events(path: Path) -> Iterator[PumpEvent]:
    """Stream PumpEvents from a JSONL recording.

    Lines that fail to parse are skipped with a warning — the script
    should never crash on malformed input, since one bad line at the
    end of a partial recording shouldn't lose the rest of the run.
    """
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
                yield PumpEvent(
                    kind=PumpInstructionKind(d["kind"]),
                    signature=d["signature"],
                    slot=int(d["slot"]),
                    block_time_unix=(
                        int(d["block_time_unix"]) if d.get("block_time_unix") is not None else None
                    ),
                    mint=d.get("mint"),
                    dev_wallet=d.get("dev_wallet"),
                    actor_wallet=d.get("actor_wallet"),
                    sol_amount=(
                        float(d["sol_amount"]) if d.get("sol_amount") is not None else None
                    ),
                    token_amount=(
                        float(d["token_amount"]) if d.get("token_amount") is not None else None
                    ),
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    "discovery.skip_malformed_line",
                    path=str(path),
                    line=line_no,
                    error=repr(exc),
                )


def _read_funder_csv(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#") or len(row) < 2:
                continue
            wallet, funder = row[0].strip(), row[1].strip()
            if wallet and funder:
                out[wallet] = funder
    return out


def _read_sell_destinations_csv(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    out: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#") or len(row) < 2:
                continue
            wallet, dest = row[0].strip(), row[1].strip()
            if wallet and dest:
                out[wallet].append(dest)
    return dict(out)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging(log_level=args.log_level, json_console=False)

    if not args.events.exists():
        logger.error("discovery.events_file_missing", path=str(args.events))
        return 2

    funder_lookup = _read_funder_csv(args.funders)
    sell_dest_lookup = _read_sell_destinations_csv(args.sell_destinations)

    cfg = KolDiscoveryConfig(top_n=args.top_n)
    events = list(_read_events(args.events))
    logger.info(
        "discovery.events_loaded",
        path=str(args.events),
        n_events=len(events),
        funders_loaded=len(funder_lookup),
        sell_destinations_wallets=len(sell_dest_lookup),
    )
    if not events:
        logger.warning("discovery.no_events", note="empty firehose recording")
        return 1

    now_unix = args.now_unix if args.now_unix is not None else int(time.time())
    kols = discover_kols(
        events,
        config=cfg,
        funder_lookup=funder_lookup,
        sell_destination_lookup=sell_dest_lookup,
        now_unix=now_unix,
    )
    n_written = write_kol_csv(kols, args.out)
    logger.info("discovery.csv_written", path=str(args.out), n_rows=n_written)
    return 0 if n_written > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
