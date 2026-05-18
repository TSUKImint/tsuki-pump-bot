"""SQLite-backed event store + append-only JSONL audit log.

Same pattern as tsuki-edge-bot, schema adapted for pump.fun:
  - `events` (generic stream) — scans, scores, fills, exits, anomalies
  - `positions` (open + closed) — entry price/size, exit ladder progress
  - `tokens` (per-token roll-up) — best score, dev wallet, age, last-seen

aiosqlite keeps writes off the event loop.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc       TEXT    NOT NULL,
    kind         TEXT    NOT NULL,
    mint         TEXT,
    dev_wallet   TEXT,
    severity     TEXT    NOT NULL DEFAULT 'info',
    summary      TEXT    NOT NULL,
    payload_json TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts       ON events(ts_utc);
CREATE INDEX IF NOT EXISTS idx_events_kind     ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_mint     ON events(mint);

CREATE TABLE IF NOT EXISTS positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mint            TEXT    NOT NULL,
    dev_wallet      TEXT,
    side            TEXT    NOT NULL DEFAULT 'BUY',
    entry_units     REAL    NOT NULL,
    entry_price_sol REAL    NOT NULL,
    entry_notional_sol REAL NOT NULL,
    opened_at_utc   TEXT    NOT NULL,
    closed_at_utc   TEXT,
    realized_pnl_sol REAL,
    status          TEXT    NOT NULL DEFAULT 'open',
    paper           INTEGER NOT NULL DEFAULT 1,
    score_at_entry  REAL,
    payload_json    TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_mint   ON positions(mint);

CREATE TABLE IF NOT EXISTS tokens (
    mint            TEXT    PRIMARY KEY,
    dev_wallet      TEXT,
    symbol          TEXT,
    name            TEXT,
    first_seen_utc  TEXT    NOT NULL,
    last_score      REAL,
    best_score      REAL,
    last_seen_utc   TEXT,
    payload_json    TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_tokens_dev ON tokens(dev_wallet);
"""


class EventStore:
    """Async event + position + token store. Use as an async context manager."""

    def __init__(self, sqlite_path: Path, audit_jsonl_path: Path) -> None:
        self.sqlite_path = sqlite_path
        self.audit_jsonl_path = audit_jsonl_path
        self._db: aiosqlite.Connection | None = None

    async def __aenter__(self) -> EventStore:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.sqlite_path)
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("EventStore not opened — use `async with`")
        return self._db

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=UTC).isoformat(timespec="microseconds")

    async def record_event(
        self,
        kind: str,
        summary: str,
        *,
        mint: str | None = None,
        dev_wallet: str | None = None,
        severity: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> int:
        ts = self._now()
        payload_json = json.dumps(payload or {}, default=str, sort_keys=True)
        cursor = await self.db.execute(
            """
            INSERT INTO events (ts_utc, kind, mint, dev_wallet, severity, summary, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, kind, mint, dev_wallet, severity, summary, payload_json),
        )
        await self.db.commit()
        row_id = int(cursor.lastrowid or 0)

        audit_line = {
            "ts_utc": ts,
            "id": row_id,
            "kind": kind,
            "mint": mint,
            "dev_wallet": dev_wallet,
            "severity": severity,
            "summary": summary,
            "payload": payload or {},
        }
        with self.audit_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(audit_line, default=str, sort_keys=True) + "\n")
        return row_id

    async def upsert_token(
        self,
        *,
        mint: str,
        dev_wallet: str | None,
        symbol: str | None,
        name: str | None,
        score: float | None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        now = self._now()
        payload_json = json.dumps(payload or {}, default=str, sort_keys=True)
        await self.db.execute(
            """
            INSERT INTO tokens (mint, dev_wallet, symbol, name, first_seen_utc,
                                last_score, best_score, last_seen_utc, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(mint) DO UPDATE SET
                dev_wallet     = COALESCE(excluded.dev_wallet, tokens.dev_wallet),
                symbol         = COALESCE(excluded.symbol, tokens.symbol),
                name           = COALESCE(excluded.name, tokens.name),
                last_score     = excluded.last_score,
                best_score     = CASE
                                    WHEN excluded.last_score IS NULL THEN tokens.best_score
                                    WHEN tokens.best_score IS NULL THEN excluded.last_score
                                    WHEN excluded.last_score > tokens.best_score
                                        THEN excluded.last_score
                                    ELSE tokens.best_score
                                 END,
                last_seen_utc  = excluded.last_seen_utc,
                payload_json   = excluded.payload_json
            """,
            (mint, dev_wallet, symbol, name, now, score, score, now, payload_json),
        )
        await self.db.commit()

    async def open_position(
        self,
        *,
        mint: str,
        dev_wallet: str | None,
        entry_units: float,
        entry_price_sol: float,
        entry_notional_sol: float,
        paper: bool,
        score_at_entry: float | None,
        payload: dict[str, Any] | None = None,
    ) -> int:
        cursor = await self.db.execute(
            """
            INSERT INTO positions
                (mint, dev_wallet, side, entry_units, entry_price_sol,
                 entry_notional_sol, opened_at_utc, status, paper,
                 score_at_entry, payload_json)
            VALUES (?, ?, 'BUY', ?, ?, ?, ?, 'open', ?, ?, ?)
            """,
            (
                mint,
                dev_wallet,
                entry_units,
                entry_price_sol,
                entry_notional_sol,
                self._now(),
                1 if paper else 0,
                score_at_entry,
                json.dumps(payload or {}, default=str, sort_keys=True),
            ),
        )
        await self.db.commit()
        return int(cursor.lastrowid or 0)

    async def close_position(
        self,
        position_id: int,
        *,
        realized_pnl_sol: float,
        status: str = "closed",
    ) -> None:
        await self.db.execute(
            """
            UPDATE positions
            SET closed_at_utc = ?, realized_pnl_sol = ?, status = ?
            WHERE id = ?
            """,
            (self._now(), realized_pnl_sol, status, position_id),
        )
        await self.db.commit()

    async def list_open_positions(self) -> list[dict[str, Any]]:
        cursor = await self.db.execute("SELECT * FROM positions WHERE status='open' ORDER BY id")
        rows = await cursor.fetchall()
        col_names = [d[0] for d in cursor.description]
        return [dict(zip(col_names, row, strict=True)) for row in rows]

    async def recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        cursor = await self.db.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        col_names = [d[0] for d in cursor.description]
        return [dict(zip(col_names, row, strict=True)) for row in rows]
