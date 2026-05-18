"""Tests for the firehose recorder + replayer (PR #D)."""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from tsukibot_pump.firehose import (
    FirehoseRecorder,
    FirehoseReplayer,
    ReplaySpeedMode,
    read_pump_event_jsonl,
    write_pump_event_jsonl,
)
from tsukibot_pump.solana.pump_program import PumpEvent, PumpInstructionKind

# ── synthetic event helpers ───────────────────────────────────────────────


def _event(
    kind: PumpInstructionKind = PumpInstructionKind.BUY,
    *,
    signature: str = "SIG_1",
    slot: int = 1000,
    mint: str | None = "MINT_X",
    dev_wallet: str | None = "DEV_X",
    actor_wallet: str | None = "ACT_X",
    block_time_unix: int | None = 1_700_000_000,
    sol_amount: float | None = 1.0,
    token_amount: float | None = 1000.0,
) -> PumpEvent:
    return PumpEvent(
        kind=kind,
        signature=signature,
        slot=slot,
        block_time_unix=block_time_unix,
        mint=mint,
        dev_wallet=dev_wallet,
        actor_wallet=actor_wallet,
        sol_amount=sol_amount,
        token_amount=token_amount,
    )


class _FakeScout:
    """In-memory scout that yields a fixed list of events once."""

    def __init__(self, events: list[PumpEvent]) -> None:
        self._events = events

    async def stream(self) -> AsyncIterator[PumpEvent]:
        for event in self._events:
            yield event


# ── recorder ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recorder_writes_jsonl_and_passes_events_through(tmp_path: Path) -> None:
    events = [
        _event(signature="SIG_A"),
        _event(kind=PumpInstructionKind.SELL, signature="SIG_B", slot=1001, sol_amount=2.5),
    ]
    scout = _FakeScout(events)
    out_path = tmp_path / "rec.jsonl"
    recorder = FirehoseRecorder(scout, output_path=out_path)

    seen: list[PumpEvent] = []
    async for ev in recorder.stream():
        seen.append(ev)

    # Same events in same order are still yielded to consumers.
    assert [e.signature for e in seen] == ["SIG_A", "SIG_B"]
    assert recorder.events_recorded == 2

    # File contains a valid JSON line per event.
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["signature"] == "SIG_A"
    assert parsed[0]["kind"] == "buy"
    assert parsed[1]["kind"] == "sell"
    assert parsed[1]["sol_amount"] == 2.5


def test_write_pump_event_jsonl_round_trip() -> None:
    """The pure-functional writer round-trips through the reader."""
    buf = io.StringIO()
    src = _event(signature="SIG_RT", slot=42)
    write_pump_event_jsonl(src, buf, recorded_at_unix=12345)

    line = buf.getvalue()
    parsed = json.loads(line)
    assert parsed["signature"] == "SIG_RT"
    assert parsed["recorded_at_unix"] == 12345


@pytest.mark.asyncio
async def test_recorder_creates_parent_directory(tmp_path: Path) -> None:
    out_path = tmp_path / "deep" / "nested" / "rec.jsonl"
    scout = _FakeScout([_event(signature="SIG_NESTED")])
    recorder = FirehoseRecorder(scout, output_path=out_path)
    async for _ev in recorder.stream():
        pass
    assert out_path.exists()


@pytest.mark.asyncio
async def test_recorder_appends_to_existing_file(tmp_path: Path) -> None:
    out_path = tmp_path / "rec.jsonl"
    out_path.write_text('{"existing": true}\n', encoding="utf-8")

    scout = _FakeScout([_event(signature="SIG_APPEND")])
    recorder = FirehoseRecorder(scout, output_path=out_path)
    async for _ev in recorder.stream():
        pass

    lines = out_path.read_text(encoding="utf-8").splitlines()
    # The original line is preserved; new event appended below.
    assert lines[0] == '{"existing": true}'
    assert json.loads(lines[1])["signature"] == "SIG_APPEND"


# ── replayer ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_replayer_yields_recorded_events_in_order(tmp_path: Path) -> None:
    rec_path = tmp_path / "rec.jsonl"
    scout = _FakeScout(
        [
            _event(signature="SIG_1", slot=1),
            _event(kind=PumpInstructionKind.SELL, signature="SIG_2", slot=2),
            _event(kind=PumpInstructionKind.CREATE, signature="SIG_3", slot=3),
        ]
    )
    async for _ in FirehoseRecorder(scout, output_path=rec_path).stream():
        pass

    replayer = FirehoseReplayer(rec_path, mode=ReplaySpeedMode.ASAP)
    replayed = [ev async for ev in replayer.stream()]
    assert [e.signature for e in replayed] == ["SIG_1", "SIG_2", "SIG_3"]
    assert [e.kind for e in replayed] == [
        PumpInstructionKind.BUY,
        PumpInstructionKind.SELL,
        PumpInstructionKind.CREATE,
    ]
    assert replayer.stats.events_emitted == 3


@pytest.mark.asyncio
async def test_replayer_handles_missing_file(tmp_path: Path) -> None:
    replayer = FirehoseReplayer(tmp_path / "does_not_exist.jsonl")
    out = [ev async for ev in replayer.stream()]
    assert out == []


@pytest.mark.asyncio
async def test_replayer_skips_malformed_lines(tmp_path: Path) -> None:
    rec_path = tmp_path / "rec.jsonl"
    # Two valid lines + one corrupted line + one with missing required field.
    valid_a = json.dumps(
        {
            "kind": "buy",
            "signature": "SIG_A",
            "slot": 1,
            "block_time_unix": None,
            "mint": "M",
            "dev_wallet": None,
            "actor_wallet": None,
            "sol_amount": None,
            "token_amount": None,
            "recorded_at_unix": 100,
        }
    )
    valid_b = json.dumps(
        {
            "kind": "sell",
            "signature": "SIG_B",
            "slot": 2,
            "block_time_unix": None,
            "mint": "M",
            "dev_wallet": None,
            "actor_wallet": None,
            "sol_amount": None,
            "token_amount": None,
            "recorded_at_unix": 100,
        }
    )
    rec_path.write_text(
        valid_a + "\n" + "not-a-json-line\n" + '{"kind": "buy"}\n' + valid_b + "\n",
        encoding="utf-8",
    )
    replayer = FirehoseReplayer(rec_path, mode=ReplaySpeedMode.ASAP)
    out = [ev async for ev in replayer.stream()]
    assert [e.signature for e in out] == ["SIG_A", "SIG_B"]


@pytest.mark.asyncio
async def test_replayer_compressed_speed_sleeps_proportionally(tmp_path: Path) -> None:
    rec_path = tmp_path / "rec.jsonl"
    # 60s gap between events — at 60x compression that should be ~1s sleep.
    rows = [
        {
            "kind": "buy",
            "signature": "SIG_1",
            "slot": 1,
            "block_time_unix": 1000,
            "mint": "M",
            "dev_wallet": None,
            "actor_wallet": None,
            "sol_amount": None,
            "token_amount": None,
            "recorded_at_unix": 1000,
        },
        {
            "kind": "buy",
            "signature": "SIG_2",
            "slot": 2,
            "block_time_unix": 1060,
            "mint": "M",
            "dev_wallet": None,
            "actor_wallet": None,
            "sol_amount": None,
            "token_amount": None,
            "recorded_at_unix": 1060,
        },
    ]
    rec_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    replayer = FirehoseReplayer(
        rec_path,
        mode=ReplaySpeedMode.COMPRESSED,
        speed_multiplier=600.0,  # 60s real → 0.1s replay sleep
        max_gap_seconds=2.0,
    )
    start = asyncio.get_event_loop().time()
    out = [ev async for ev in replayer.stream()]
    elapsed = asyncio.get_event_loop().time() - start
    assert [e.signature for e in out] == ["SIG_1", "SIG_2"]
    # Should sleep ~0.1s; allow generous bounds for CI jitter.
    assert 0.05 <= elapsed <= 1.0


@pytest.mark.asyncio
async def test_replayer_stop_event_short_circuits_long_gap(tmp_path: Path) -> None:
    rec_path = tmp_path / "rec.jsonl"
    rows = [
        {
            "kind": "buy",
            "signature": "SIG_1",
            "slot": 1,
            "block_time_unix": 1000,
            "mint": "M",
            "dev_wallet": None,
            "actor_wallet": None,
            "sol_amount": None,
            "token_amount": None,
            "recorded_at_unix": 1000,
        },
        {
            "kind": "buy",
            "signature": "SIG_2",
            "slot": 2,
            "block_time_unix": 2000,
            "mint": "M",
            "dev_wallet": None,
            "actor_wallet": None,
            "sol_amount": None,
            "token_amount": None,
            "recorded_at_unix": 2000,
        },
    ]
    rec_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    stop = asyncio.Event()
    replayer = FirehoseReplayer(
        rec_path,
        mode=ReplaySpeedMode.REALTIME,
        max_gap_seconds=30.0,  # 1000s gap would normally sleep ~30s
        stop_event=stop,
    )

    async def _drain() -> list[PumpEvent]:
        return [ev async for ev in replayer.stream()]

    drain_task = asyncio.create_task(_drain())
    await asyncio.sleep(0.05)  # let the first event emit + start sleeping
    stop.set()
    out = await asyncio.wait_for(drain_task, timeout=2.0)
    # Only the first event was emitted before stop fired.
    assert [e.signature for e in out] == ["SIG_1"]


def test_read_pump_event_jsonl_yields_tuples(tmp_path: Path) -> None:
    rec_path = tmp_path / "rec.jsonl"
    rec_path.write_text(
        json.dumps(
            {
                "kind": "buy",
                "signature": "SIG_R",
                "slot": 9,
                "block_time_unix": 1234,
                "mint": "M",
                "dev_wallet": None,
                "actor_wallet": None,
                "sol_amount": None,
                "token_amount": None,
                "recorded_at_unix": 5678,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pairs = list(read_pump_event_jsonl(rec_path))
    assert len(pairs) == 1
    event, recorded_at = pairs[0]
    assert event.signature == "SIG_R"
    assert recorded_at == 5678


def test_replay_speed_mode_rejects_zero_multiplier() -> None:
    with pytest.raises(ValueError):
        FirehoseReplayer(Path("/tmp/does_not_matter.jsonl"), speed_multiplier=0.0)


def test_replay_speed_mode_rejects_negative_multiplier() -> None:
    with pytest.raises(ValueError):
        FirehoseReplayer(Path("/tmp/does_not_matter.jsonl"), speed_multiplier=-1.0)


# ── end-to-end via the orchestrator ───────────────────────────────────────


@pytest.mark.asyncio
async def test_replayer_duck_types_into_run_scout_loop(tmp_path: Path) -> None:
    """The replayer must drop into ``run_scout_loop`` as a scout substitute."""
    from unittest.mock import MagicMock

    from tsukibot_pump.orchestrator import run_scout_loop

    # Write three events to a recording.
    rec_path = tmp_path / "rec.jsonl"
    scout = _FakeScout(
        [
            _event(signature="SIG_E2E_1"),
            _event(signature="SIG_E2E_2", kind=PumpInstructionKind.SELL),
            _event(signature="SIG_E2E_3", kind=PumpInstructionKind.CREATE),
        ]
    )
    async for _ in FirehoseRecorder(scout, output_path=rec_path).stream():
        pass

    # Build a fake orchestrator context — only the bits run_scout_loop touches.
    ingested: list[PumpEvent] = []
    aggregator = MagicMock()
    aggregator.ingest.side_effect = ingested.append
    kill = MagicMock()
    kill.tripped = False
    ctx = MagicMock(aggregator=aggregator, kill=kill)

    replayer = FirehoseReplayer(rec_path, mode=ReplaySpeedMode.ASAP)
    await run_scout_loop(ctx, replayer)
    assert [e.signature for e in ingested] == ["SIG_E2E_1", "SIG_E2E_2", "SIG_E2E_3"]


# ── CLI integration ───────────────────────────────────────────────────────


def test_firehose_cli_replay_command(tmp_path: Path) -> None:
    """Subprocess-based CLI test to avoid stdout-capture interference.

    structlog's PrintLogger caches the stdout it was first configured with,
    so once another test wraps stdout via ``capsys`` and tears it down, a
    subsequent in-process ``main()`` call will write to a closed file. Running
    the CLI in a subprocess sidesteps the issue entirely.
    """
    import subprocess
    import sys

    rec_path = tmp_path / "rec.jsonl"
    row = {
        "kind": "buy",
        "signature": "SIG_CLI",
        "slot": 7,
        "block_time_unix": 1234,
        "mint": "MINT_CLI",
        "dev_wallet": None,
        "actor_wallet": "ACTOR_CLI",
        "sol_amount": 0.5,
        "token_amount": None,
        "recorded_at_unix": 5678,
    }
    rec_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tsukibot_pump.firehose",
            "--log-level",
            "ERROR",
            "replay",
            "--events",
            str(rec_path),
            "--speed",
            "asap",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "MINT_CLI"[:8] in proc.stdout
    assert "ACTOR_CL" in proc.stdout


def test_firehose_cli_replay_returns_two_if_missing(tmp_path: Path) -> None:
    """Subprocess test for the missing-file path (see sibling test above)."""
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tsukibot_pump.firehose",
            "--log-level",
            "ERROR",
            "replay",
            "--events",
            str(tmp_path / "nope.jsonl"),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 2
