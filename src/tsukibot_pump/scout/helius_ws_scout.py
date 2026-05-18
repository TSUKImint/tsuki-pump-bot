"""Helius WebSocket logsSubscribe transport for the pump.fun firehose.

Why this exists
---------------
The v0.2 `PumpScout` polls `getSignaturesForAddress` over HTTP every N
seconds (default 5 s). That's free and works against any RPC, but it caps
detection latency at the poll interval — too slow for the v0.3 "early
conviction" lane which wants to react inside the first 30 s of CREATE.

Helius supports a standard JSON-RPC method, `logsSubscribe`, on every plan
(including the free tier) at:

    wss://mainnet.helius-rpc.com/?api-key=<KEY>

Latency reported by Helius: ~200 ms end-to-end (well under the polling
floor). See https://www.helius.dev/docs/data-streaming/websocket-and-webhooks
for the official latency claim.

We filter by `mentions: [PUMP_FUN_PROGRAM_ID]` so we only see logs that
touched the pump.fun program. The `value.signature` field is then fed into
the same `get_transaction` / `parse_pump_instructions` pipeline used by
the polling path — so downstream code doesn't change.

The transport is conservative:
  - On any WS error (connection drop, parse failure, server reject) we log,
    bail out of the subscription, and let the caller fall back to HTTP.
  - We never block the caller for more than `connect_timeout` on initial
    connect.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import structlog

from ..solana.pump_program import (
    PumpEvent,
    PumpInstructionKind,
    parse_pump_instructions,
    sol_delta_for_wallet,
)
from ..solana.rpc import PUMP_FUN_PROGRAM_ID, SolanaRPCClient

logger = structlog.get_logger(__name__)


@dataclass
class HeliusScoutStats:
    """Stats parallel to ScoutStats — used by dashboards."""

    signatures_seen: int = 0
    transactions_parsed: int = 0
    create_events: int = 0
    buy_events: int = 0
    sell_events: int = 0
    rpc_errors: int = 0
    last_signature: str = ""
    connected_at_unix: float | None = None
    ws_reconnects: int = 0


class HeliusWebsocketScout:
    """logsSubscribe-backed pump.fun firehose.

    Iterator surface is identical to PumpScout so the orchestrator can swap
    them in via duck-typing.
    """

    def __init__(
        self,
        rpc: SolanaRPCClient,
        ws_url: str,
        *,
        connect_timeout_seconds: float = 10.0,
        ping_interval_seconds: float = 20.0,
        reconnect_max_backoff_seconds: float = 60.0,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        if not ws_url:
            raise ValueError("ws_url must be non-empty")
        if not ws_url.startswith(("ws://", "wss://")):
            raise ValueError(f"ws_url must be ws:// or wss://, got {ws_url[:8]}")
        self.rpc = rpc
        self.ws_url = ws_url
        self.connect_timeout_seconds = connect_timeout_seconds
        self.ping_interval_seconds = ping_interval_seconds
        self.reconnect_max_backoff_seconds = reconnect_max_backoff_seconds
        self.stop_event = stop_event or asyncio.Event()
        self.stats = HeliusScoutStats()
        self._seen_signatures: set[str] = set()
        self._max_seen_signatures = 5_000

    async def stream(self) -> AsyncIterator[PumpEvent]:
        """Yield pump.fun PumpEvents from the Helius WS firehose.

        Reconnects with exponential backoff on transport failure. Stops
        when `stop_event` is set.
        """
        backoff = 1.0
        while not self.stop_event.is_set():
            try:
                async for event in self._stream_one_session():
                    yield event
                    if self.stop_event.is_set():
                        return
                # Clean exit (stop_event tripped during yield).
                if self.stop_event.is_set():
                    return
                backoff = 1.0  # any clean cycle resets backoff
            except Exception as exc:
                self.stats.rpc_errors += 1
                self.stats.ws_reconnects += 1
                logger.warning(
                    "helius_ws.session_error",
                    err=str(exc),
                    backoff=backoff,
                )
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self.stop_event.wait(), timeout=backoff)
                backoff = min(self.reconnect_max_backoff_seconds, backoff * 2)

    async def _stream_one_session(self) -> AsyncIterator[PumpEvent]:
        """Open one WS session and yield events until it closes / errors."""
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - hard dep
            raise RuntimeError("websockets is not installed; install tsuki-pump-bot deps") from exc

        async with await asyncio.wait_for(
            websockets.connect(
                self.ws_url,
                ping_interval=self.ping_interval_seconds,
                max_size=2**20,  # 1 MiB ought to be enough for any log frame
                open_timeout=self.connect_timeout_seconds,
            ),
            timeout=self.connect_timeout_seconds + 5,
        ) as ws:
            self.stats.connected_at_unix = asyncio.get_running_loop().time()
            # Subscribe to logs that mention the pump.fun program.
            sub = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "logsSubscribe",
                "params": [
                    {"mentions": [PUMP_FUN_PROGRAM_ID]},
                    {"commitment": "confirmed"},
                ],
            }
            await ws.send(json.dumps(sub))
            logger.info("helius_ws.subscribed", url_host=self.ws_url.split("?", 1)[0])

            async for raw in ws:
                if self.stop_event.is_set():
                    return
                try:
                    message = json.loads(raw) if isinstance(raw, str) else None
                except (ValueError, TypeError):
                    continue
                if not isinstance(message, dict):
                    continue
                # Ignore the subscription-confirmation reply.
                if "result" in message and "params" not in message:
                    continue
                params = message.get("params")
                if not isinstance(params, dict):
                    continue
                result = params.get("result")
                if not isinstance(result, dict):
                    continue
                value = result.get("value")
                if not isinstance(value, dict):
                    continue

                signature = value.get("signature")
                err = value.get("err")
                if not isinstance(signature, str) or err is not None:
                    continue
                # Per-session dedupe to handle Helius's occasional repeats.
                if signature in self._seen_signatures:
                    continue
                if len(self._seen_signatures) >= self._max_seen_signatures:
                    # Bounded set; cheap eviction by dropping the whole set.
                    self._seen_signatures.clear()
                self._seen_signatures.add(signature)

                self.stats.signatures_seen += 1
                self.stats.last_signature = signature

                tx = await self.rpc.get_transaction(signature)
                if not tx:
                    continue
                self.stats.transactions_parsed += 1

                events = parse_pump_instructions(tx)
                for event in events:
                    enriched = self._enrich_amounts(event, tx)
                    if enriched.kind == PumpInstructionKind.CREATE:
                        self.stats.create_events += 1
                    elif enriched.kind == PumpInstructionKind.BUY:
                        self.stats.buy_events += 1
                    elif enriched.kind == PumpInstructionKind.SELL:
                        self.stats.sell_events += 1
                    yield enriched

    @staticmethod
    def _enrich_amounts(event: PumpEvent, tx: dict[str, object]) -> PumpEvent:
        """Fill in SOL amount via balance diff if available. Same as PumpScout."""
        if event.actor_wallet is None:
            return event
        delta = sol_delta_for_wallet(tx, event.actor_wallet)
        if delta is None:
            return event
        return PumpEvent(
            kind=event.kind,
            signature=event.signature,
            slot=event.slot,
            block_time_unix=event.block_time_unix,
            mint=event.mint,
            dev_wallet=event.dev_wallet,
            actor_wallet=event.actor_wallet,
            sol_amount=abs(delta),
            token_amount=event.token_amount,
        )
