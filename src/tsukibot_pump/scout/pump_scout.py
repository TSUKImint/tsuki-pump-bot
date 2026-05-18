"""Pump.fun program firehose.

Two transport modes:

  * **gRPC** (Helius Yellowstone / Triton / Shyft) — preferred. Sub-100ms
    signature stream. We deliberately don't bundle the gRPC client in
    `pyproject.toml` to avoid heavy deps for paper-mock users; the
    `_grpc_subscribe` method imports `grpc` lazily and surfaces a clear
    error if it's missing.

  * **HTTP polling** — fallback. Calls
    `getSignaturesForAddress(PUMP_FUN_PROGRAM_ID)` every
    `http_poll_interval_seconds` and diffs against the last seen signature.
    Latency 1-5s, free, works on any RPC.

Either transport produces a stream of `PumpEvent`s consumed by the
`TokenStateAggregator`.
"""

from __future__ import annotations

import asyncio
import contextlib
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
class ScoutStats:
    signatures_seen: int = 0
    transactions_parsed: int = 0
    create_events: int = 0
    buy_events: int = 0
    sell_events: int = 0
    rpc_errors: int = 0
    last_signature: str = ""


class PumpScout:
    """HTTP-polling firehose for the pump.fun program.

    Use as an async iterator:

        async with SolanaRPCClient(...) as rpc:
            scout = PumpScout(rpc, poll_interval_seconds=5)
            async for event in scout.stream():
                ...
    """

    def __init__(
        self,
        rpc: SolanaRPCClient,
        *,
        poll_interval_seconds: float = 5.0,
        max_signatures_per_poll: int = 100,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        self.rpc = rpc
        self.poll_interval_seconds = poll_interval_seconds
        self.max_signatures_per_poll = max(1, min(max_signatures_per_poll, 1000))
        self.stop_event = stop_event or asyncio.Event()
        self.stats = ScoutStats()
        self._last_signature: str | None = None

    async def stream(self) -> AsyncIterator[PumpEvent]:
        """Yield pump.fun program events until `stop_event` is set."""
        while not self.stop_event.is_set():
            try:
                async for event in self._poll_once():
                    yield event
            except Exception as exc:
                self.stats.rpc_errors += 1
                logger.warning("pump_scout.poll_error", err=str(exc))

            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self.stop_event.wait(),
                    timeout=self.poll_interval_seconds,
                )

    async def _poll_once(self) -> AsyncIterator[PumpEvent]:
        sigs = await self.rpc.get_signatures_for_address(
            PUMP_FUN_PROGRAM_ID,
            limit=self.max_signatures_per_poll,
            until=self._last_signature,
        )
        if not sigs:
            return
        # API returns newest-first; process oldest-first for monotonic state.
        sigs.reverse()
        self.stats.signatures_seen += len(sigs)
        for entry in sigs:
            sig = entry.get("signature")
            if not sig:
                continue
            if entry.get("err") is not None:
                # Failed transactions are noise.
                continue
            tx = await self.rpc.get_transaction(sig)
            if not tx:
                continue
            self.stats.transactions_parsed += 1
            events = parse_pump_instructions(tx)
            for event in events:
                event = self._enrich_amounts(event, tx)
                if event.kind == PumpInstructionKind.CREATE:
                    self.stats.create_events += 1
                elif event.kind == PumpInstructionKind.BUY:
                    self.stats.buy_events += 1
                elif event.kind == PumpInstructionKind.SELL:
                    self.stats.sell_events += 1
                yield event
            self._last_signature = sig
            self.stats.last_signature = sig

    @staticmethod
    def _enrich_amounts(event: PumpEvent, tx: dict[str, object]) -> PumpEvent:
        """Fill in SOL amount via balance diff if available.

        We don't try to compute token deltas here — that requires parsing
        SPL token balances which is more brittle. The aggregator approximates
        token-units-bought via the curve math instead.
        """
        if event.actor_wallet is None:
            return event
        delta = sol_delta_for_wallet(tx, event.actor_wallet)
        if delta is None:
            return event
        # Buyers spend SOL (negative delta); sellers receive (positive).
        # Report absolute value as `sol_amount`.
        return PumpEvent(
            kind=event.kind,
            signature=event.signature,
            slot=event.slot,
            block_time_unix=event.block_time_unix,
            mint=event.mint,
            dev_wallet=event.dev_wallet,
            actor_wallet=event.actor_wallet,
            sol_amount=abs(delta) if delta is not None else None,
            token_amount=event.token_amount,
        )
