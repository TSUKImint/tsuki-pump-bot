"""Record + replay the pump.fun firehose for offline backtesting.

Two pieces:

* :class:`FirehoseRecorder` — wraps any scout-shaped object
  (``.stream() -> AsyncIterator[PumpEvent]``) and writes every event
  passing through to a JSONL file. Transparent middleware: the
  consumer still sees the same events in the same order.
* :class:`FirehoseReplayer` — duck-typed to :class:`PumpScout`. Reads
  a JSONL recording and re-emits events with an optional speed
  multiplier so the orchestrator can backtest a session deterministically
  without an RPC connection.

The JSONL schema matches what
:mod:`tsukibot_pump.discovery.__main__` already parses, so a single
recording feeds both the backtester and the KOL discovery pipeline.
"""

from .recorder import FirehoseRecorder, write_pump_event_jsonl
from .replayer import FirehoseReplayer, ReplaySpeedMode, read_pump_event_jsonl

__all__ = [
    "FirehoseRecorder",
    "FirehoseReplayer",
    "ReplaySpeedMode",
    "read_pump_event_jsonl",
    "write_pump_event_jsonl",
]
