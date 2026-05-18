"""Record a live :class:`PumpEvent` stream to JSONL for later replay.

The recorder is **transparent middleware** — it wraps any
scout-shaped object that exposes
``stream() -> AsyncIterator[PumpEvent]`` and re-emits every event,
writing a JSON line per event to disk along the way. So you can
record while paper-trading simultaneously, no second RPC connection
needed.

JSON schema (one event per line, in the order observed):

    {
        "kind": "create" | "buy" | "sell" | ...,
        "signature": "<base58 sig>",
        "slot": 1234567,
        "block_time_unix": 1715000000 | null,
        "mint": "<base58 mint>" | null,
        "dev_wallet": "<base58>" | null,
        "actor_wallet": "<base58>" | null,
        "sol_amount": 1.234 | null,
        "token_amount": 1000.0 | null,
        "recorded_at_unix": 1715000123  // wall-clock when written
    }

``recorded_at_unix`` is added by the recorder, not by the scout, so
the replayer can compute wall-clock inter-event gaps even when
``block_time_unix`` is missing (which it sometimes is on
older/unconfirmed transactions).
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

import structlog

from ..solana.pump_program import PumpEvent

logger = structlog.get_logger(__name__)


class _ScoutLike(Protocol):
    """Duck-typed protocol so the recorder works with any scout."""

    def stream(self) -> AsyncIterator[PumpEvent]: ...


def write_pump_event_jsonl(
    event: PumpEvent, sink: object, *, recorded_at_unix: int | None = None
) -> None:
    """Write one event as JSON to a text-mode file-like ``sink``.

    Separated from :class:`FirehoseRecorder` so the CLI can call it
    directly when streaming from a pre-built iterable in tests.
    """
    row = {
        "kind": event.kind.value,
        "signature": event.signature,
        "slot": event.slot,
        "block_time_unix": event.block_time_unix,
        "mint": event.mint,
        "dev_wallet": event.dev_wallet,
        "actor_wallet": event.actor_wallet,
        "sol_amount": event.sol_amount,
        "token_amount": event.token_amount,
        "recorded_at_unix": recorded_at_unix if recorded_at_unix is not None else int(time.time()),
    }
    write = getattr(sink, "write", None)
    if write is None:
        raise TypeError("sink must be a file-like object with `.write`")
    write(json.dumps(row, sort_keys=True) + "\n")
    flush = getattr(sink, "flush", None)
    if callable(flush):
        flush()


class FirehoseRecorder:
    """Wraps an existing scout to also record events to a JSONL file.

    Usage::

        scout = PumpScout(rpc, poll_interval_seconds=5.0)
        recorder = FirehoseRecorder(scout, output_path=Path("state/firehose.jsonl"))
        async for event in recorder.stream():
            aggregator.ingest(event)

    The recorder owns the output file: opens on first event, closes
    when :meth:`close` is called or the wrapper is garbage-collected.
    """

    def __init__(self, scout: _ScoutLike, *, output_path: Path) -> None:
        self._scout = scout
        self._output_path = output_path
        self._fh: object | None = None
        self._events_recorded = 0

    @property
    def events_recorded(self) -> int:
        return self._events_recorded

    @property
    def output_path(self) -> Path:
        return self._output_path

    async def stream(self) -> AsyncIterator[PumpEvent]:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        # Append mode — if the recording is interrupted and resumed,
        # the existing file is preserved.
        self._fh = self._output_path.open("a", encoding="utf-8")
        try:
            async for event in self._scout.stream():
                try:
                    write_pump_event_jsonl(event, self._fh)
                    self._events_recorded += 1
                except OSError as exc:
                    logger.warning(
                        "firehose_recorder.write_failed",
                        path=str(self._output_path),
                        error=repr(exc),
                    )
                yield event
        finally:
            self.close()

    def close(self) -> None:
        if self._fh is not None:
            close = getattr(self._fh, "close", None)
            if callable(close):
                close()
            self._fh = None
