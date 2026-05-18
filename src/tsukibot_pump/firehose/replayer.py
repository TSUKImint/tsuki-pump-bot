"""Replay a recorded JSONL firehose as an :class:`AsyncIterator`.

The replayer is **duck-typed to PumpScout** — it exposes the same
``stream()`` method so ``run_scout_loop`` can consume from it
unchanged, without needing an RPC connection at all.

Three speed modes:

* ``"realtime"`` — sleep between events to match the recorded
  inter-event gap. Best for verifying the bot's behaviour with the
  same temporal distribution it saw live.
* ``"compressed"`` — multiply gaps by ``speed_multiplier``. Useful
  for "replay a 1h session in 6 minutes" debugging.
* ``"asap"`` — yield events as fast as the consumer can pull them.
  Best for the deterministic backtest path where wall-clock latency
  is irrelevant.

In every mode, the replayer is **deterministic** with respect to
event order — the JSONL file dictates everything. Combined with
:attr:`PaperRealismConfig.rng_seed` from PR #2, this gives bit-for-bit
reproducible backtests: same firehose + same seed → same fills →
same P&L.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import structlog

from ..solana.pump_program import PumpEvent, PumpInstructionKind

logger = structlog.get_logger(__name__)


class ReplaySpeedMode(StrEnum):
    REALTIME = "realtime"
    COMPRESSED = "compressed"
    ASAP = "asap"


@dataclass
class ReplayStats:
    events_emitted: int = 0
    malformed_lines: int = 0
    last_signature: str = ""


def read_pump_event_jsonl(path: Path) -> Iterator[tuple[PumpEvent, int | None]]:
    """Yield ``(event, recorded_at_unix)`` pairs from a JSONL recording.

    ``recorded_at_unix`` is whatever the recorder wrote, which may be
    ``None`` for events parsed from a third-party dataset that
    doesn't carry one. The replayer falls back to ``block_time_unix``
    in that case.
    """
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
                event = PumpEvent(
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
                recorded_at = d.get("recorded_at_unix")
                if recorded_at is not None:
                    recorded_at = int(recorded_at)
                yield event, recorded_at
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    "firehose_replayer.skip_malformed_line",
                    path=str(path),
                    line=line_no,
                    error=repr(exc),
                )


class FirehoseReplayer:
    """Replay a JSONL firehose recording.

    Usage::

        replayer = FirehoseReplayer(Path("state/firehose.jsonl"), mode="asap")
        async for event in replayer.stream():
            aggregator.ingest(event)

    The replayer is single-use: ``stream()`` reads from disk lazily,
    so calling it twice rereads the file from the beginning.
    """

    def __init__(
        self,
        path: Path,
        *,
        mode: ReplaySpeedMode | str = ReplaySpeedMode.ASAP,
        speed_multiplier: float = 60.0,
        stop_event: asyncio.Event | None = None,
        max_gap_seconds: float = 30.0,
    ) -> None:
        self._path = path
        self._mode = ReplaySpeedMode(mode) if not isinstance(mode, ReplaySpeedMode) else mode
        if speed_multiplier <= 0:
            raise ValueError("speed_multiplier must be > 0")
        self._speed_multiplier = speed_multiplier
        self._max_gap_seconds = max(0.0, max_gap_seconds)
        self.stop_event = stop_event or asyncio.Event()
        self.stats = ReplayStats()

    @property
    def mode(self) -> ReplaySpeedMode:
        return self._mode

    @property
    def path(self) -> Path:
        return self._path

    async def stream(self) -> AsyncIterator[PumpEvent]:
        if not self._path.exists():
            logger.error("firehose_replayer.file_missing", path=str(self._path))
            return

        last_event_time: int | None = None
        for event, recorded_at in read_pump_event_jsonl(self._path):
            if self.stop_event.is_set():
                break

            # Decide which timestamp drives the gap.
            current_time = recorded_at if recorded_at is not None else event.block_time_unix
            if (
                self._mode is not ReplaySpeedMode.ASAP
                and last_event_time is not None
                and current_time is not None
            ):
                gap = max(0.0, float(current_time - last_event_time))
                if self._mode is ReplaySpeedMode.COMPRESSED:
                    gap = gap / self._speed_multiplier
                gap = min(self._max_gap_seconds, gap)
                if gap > 0:
                    try:
                        await asyncio.wait_for(self.stop_event.wait(), timeout=gap)
                        # stop_event fired during the gap.
                        break
                    except TimeoutError:
                        pass

            self.stats.events_emitted += 1
            self.stats.last_signature = event.signature
            last_event_time = current_time
            yield event
