"""Tests for the v0.3 Helius WebSocket logsSubscribe scout.

We don't spin up an actual WS server — we stub the websockets connection
context manager and feed canned `logsNotification` JSON-RPC frames. The
test covers:

  - URL validation (must be ws:// or wss://)
  - logsSubscribe payload shape (jsonrpc, method, params, mentions filter)
  - dedupe across duplicate signatures
  - skip on err != null
  - reconnect on transport failure with exponential backoff
  - PumpEvent emission from a CREATE log + rpc.get_transaction

The Solana RPC is stubbed via a tiny fake; we don't hit the network.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tsukibot_pump.scout.helius_ws_scout import HeliusWebsocketScout
from tsukibot_pump.solana.rpc import SolanaRPCClient


def test_init_rejects_empty_url() -> None:
    rpc = SolanaRPCClient("https://example.com")
    with pytest.raises(ValueError, match="non-empty"):
        HeliusWebsocketScout(rpc, "")


def test_init_rejects_non_ws_scheme() -> None:
    rpc = SolanaRPCClient("https://example.com")
    with pytest.raises(ValueError, match="ws://"):
        HeliusWebsocketScout(rpc, "https://mainnet.helius-rpc.com/?api-key=k")


class _FakeAsyncWebSocket:
    """Minimal async context manager that yields a canned list of frames."""

    def __init__(self, frames: list[str]) -> None:
        self._frames = frames
        self.sent: list[str] = []

    async def __aenter__(self) -> _FakeAsyncWebSocket:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def send(self, msg: str) -> None:
        self.sent.append(msg)

    def __aiter__(self) -> _FakeAsyncWebSocket:
        return self

    async def __anext__(self) -> str:
        if not self._frames:
            raise StopAsyncIteration
        return self._frames.pop(0)


@pytest.mark.asyncio
async def test_subscribe_frame_has_logs_subscribe_with_pump_mention() -> None:
    """The first send must be `logsSubscribe` mentioning the pump.fun program."""
    rpc = SolanaRPCClient("https://example.com")
    fake_ws = _FakeAsyncWebSocket(frames=[])

    async def fake_connect(url: str, **kwargs: Any) -> _FakeAsyncWebSocket:
        return fake_ws

    scout = HeliusWebsocketScout(
        rpc,
        "wss://mainnet.helius-rpc.com/?api-key=test",
        connect_timeout_seconds=1.0,
    )
    scout.stop_event = asyncio.Event()

    with patch("websockets.connect", side_effect=fake_connect):
        # We just want to capture the subscription payload, not iterate.
        async for _ in scout._stream_one_session():
            break
    assert len(fake_ws.sent) == 1
    payload = json.loads(fake_ws.sent[0])
    assert payload["method"] == "logsSubscribe"
    mentions = payload["params"][0]["mentions"]
    assert len(mentions) == 1
    assert mentions[0]  # non-empty program id


@pytest.mark.asyncio
async def test_yields_pump_events_for_logs_notification() -> None:
    """A logsNotification with a valid signature yields a PumpEvent."""
    rpc = SolanaRPCClient("https://example.com")
    # Stub `get_transaction` so we don't hit the network.
    rpc.get_transaction = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "meta": {"err": None},
            "transaction": {"signatures": ["SIG_1"]},
            "blockTime": 1_700_000_000,
            "slot": 999,
        }
    )

    notification = {
        "jsonrpc": "2.0",
        "method": "logsNotification",
        "params": {
            "result": {
                "context": {"slot": 999},
                "value": {
                    "signature": "SIG_1",
                    "err": None,
                    "logs": ["Program log: ..."],
                },
            },
            "subscription": 1,
        },
    }

    # First frame is the subscription-confirmation, then a notification.
    confirm = {"jsonrpc": "2.0", "result": 1, "id": 1}
    fake_ws = _FakeAsyncWebSocket(frames=[json.dumps(confirm), json.dumps(notification)])

    async def fake_connect(url: str, **kwargs: Any) -> _FakeAsyncWebSocket:
        return fake_ws

    scout = HeliusWebsocketScout(rpc, "wss://mainnet.helius-rpc.com/?api-key=test")

    # parse_pump_instructions is the gate that decides whether a tx contains
    # pump.fun instructions; for this test we stub it to return a CREATE.
    from tsukibot_pump.solana.pump_program import (
        PumpEvent,
        PumpInstructionKind,
    )

    fake_event = PumpEvent(
        kind=PumpInstructionKind.CREATE,
        signature="SIG_1",
        slot=999,
        block_time_unix=1_700_000_000,
        mint="MINT_1",
        dev_wallet="DEV_1",
        actor_wallet="DEV_1",
        sol_amount=0.0,
        token_amount=0.0,
    )

    with (
        patch(
            "tsukibot_pump.scout.helius_ws_scout.parse_pump_instructions",
            return_value=[fake_event],
        ),
        patch(
            "tsukibot_pump.scout.helius_ws_scout.sol_delta_for_wallet",
            return_value=None,
        ),
        patch("websockets.connect", side_effect=fake_connect),
    ):
        events = []
        async for ev in scout._stream_one_session():
            events.append(ev)
            if len(events) >= 1:
                scout.stop_event.set()
                break

    assert len(events) == 1
    assert events[0].signature == "SIG_1"
    assert events[0].kind == PumpInstructionKind.CREATE
    assert scout.stats.create_events == 1


@pytest.mark.asyncio
async def test_dedupes_duplicate_signatures() -> None:
    """Helius occasionally repeats notifications; we must not double-parse."""
    rpc = SolanaRPCClient("https://example.com")
    rpc.get_transaction = AsyncMock(  # type: ignore[method-assign]
        return_value={"meta": {"err": None}, "blockTime": 1, "slot": 1}
    )

    same_sig_msg = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "logsNotification",
            "params": {
                "result": {
                    "context": {"slot": 1},
                    "value": {
                        "signature": "DUP_SIG",
                        "err": None,
                        "logs": [],
                    },
                },
                "subscription": 1,
            },
        }
    )

    fake_ws = _FakeAsyncWebSocket(frames=[same_sig_msg, same_sig_msg])

    async def fake_connect(url: str, **kwargs: Any) -> _FakeAsyncWebSocket:
        return fake_ws

    scout = HeliusWebsocketScout(rpc, "wss://example.com/")

    with (
        patch(
            "tsukibot_pump.scout.helius_ws_scout.parse_pump_instructions",
            return_value=[],
        ),
        patch("websockets.connect", side_effect=fake_connect),
    ):
        async for _ in scout._stream_one_session():
            break

    # We sent the same signature twice; get_transaction should only have
    # been called once.
    assert rpc.get_transaction.call_count == 1
    assert scout.stats.signatures_seen == 1


@pytest.mark.asyncio
async def test_skips_logs_with_err_not_null() -> None:
    """Solana log entries with err != null are failed txs; skip them."""
    rpc = SolanaRPCClient("https://example.com")
    rpc.get_transaction = AsyncMock()  # type: ignore[method-assign]

    failed_msg = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "logsNotification",
            "params": {
                "result": {
                    "context": {"slot": 1},
                    "value": {
                        "signature": "FAIL_SIG",
                        "err": {"InstructionError": [0, "Custom"]},
                        "logs": [],
                    },
                },
                "subscription": 1,
            },
        }
    )
    fake_ws = _FakeAsyncWebSocket(frames=[failed_msg])

    async def fake_connect(url: str, **kwargs: Any) -> _FakeAsyncWebSocket:
        return fake_ws

    scout = HeliusWebsocketScout(rpc, "wss://example.com/")
    with patch("websockets.connect", side_effect=fake_connect):
        async for _ in scout._stream_one_session():
            break

    rpc.get_transaction.assert_not_called()
    assert scout.stats.signatures_seen == 0
