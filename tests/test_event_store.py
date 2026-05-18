"""EventStore tests."""

from __future__ import annotations

import json
from pathlib import Path

from tsukibot_pump.core.event_store import EventStore


async def test_event_store_persists_and_audits(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    audit_path = tmp_path / "audit.jsonl"
    async with EventStore(db_path, audit_path) as store:
        eid = await store.record_event(
            kind="test.event",
            summary="hello",
            mint="MINT1",
            dev_wallet="DEV1",
            severity="info",
            payload={"x": 1},
        )
        assert eid > 0

    # Re-open and verify persistence.
    async with EventStore(db_path, audit_path) as store2:
        recent = await store2.recent_events()
        assert len(recent) == 1
        assert recent[0]["kind"] == "test.event"
        assert recent[0]["mint"] == "MINT1"

    # Audit file is a single JSON line.
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["kind"] == "test.event"
    assert parsed["payload"] == {"x": 1}


async def test_upsert_token_tracks_best_score(tmp_path: Path) -> None:
    async with EventStore(tmp_path / "e.db", tmp_path / "a.jsonl") as store:
        await store.upsert_token(
            mint="M",
            dev_wallet="D",
            symbol="ABC",
            name="Abc",
            score=50.0,
        )
        await store.upsert_token(
            mint="M",
            dev_wallet="D",
            symbol="ABC",
            name="Abc",
            score=70.0,
        )
        # Lower score afterwards should not overwrite best_score.
        await store.upsert_token(
            mint="M",
            dev_wallet="D",
            symbol="ABC",
            name="Abc",
            score=40.0,
        )
        cursor = await store.db.execute("SELECT last_score, best_score FROM tokens WHERE mint='M'")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 40.0
        assert row[1] == 70.0


async def test_position_lifecycle(tmp_path: Path) -> None:
    async with EventStore(tmp_path / "e.db", tmp_path / "a.jsonl") as store:
        pid = await store.open_position(
            mint="MINT",
            dev_wallet="DEV",
            entry_units=100.0,
            entry_price_sol=0.0001,
            entry_notional_sol=0.01,
            paper=True,
            score_at_entry=75.0,
            payload={"foo": "bar"},
        )
        assert pid > 0
        open_rows = await store.list_open_positions()
        assert len(open_rows) == 1
        assert open_rows[0]["mint"] == "MINT"
        assert open_rows[0]["status"] == "open"

        await store.close_position(pid, realized_pnl_sol=0.05)
        open_rows = await store.list_open_positions()
        assert open_rows == []
