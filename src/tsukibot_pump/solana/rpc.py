"""Minimal async Solana JSON-RPC client over httpx.

We only need a small surface area for paper modes:
  - getSignaturesForAddress (firehose for the pump.fun program)
  - getTransaction (parse buys / sells)
  - getAccountInfo (read bonding curve state)
  - getSlot / getBlockTime (timing)

This avoids pulling in the heavyweight `solana-py` SDK for read paths.
Rate-limited via a token-bucket so we play nicely with free RPCs.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

PUMP_FUN_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
"""pump.fun program ID on mainnet."""

PUMPSWAP_PROGRAM_ID = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
"""PumpSwap AMM program ID (graduated tokens migrate here, since March 2025)."""


class RPCError(Exception):
    """Raised for non-recoverable RPC errors (4xx / parse failures)."""


class TokenBucket:
    """Simple async token-bucket rate limiter."""

    def __init__(self, rate_per_second: float, capacity: float | None = None) -> None:
        self._rate = rate_per_second
        self._capacity = capacity if capacity is not None else rate_per_second
        self._tokens = self._capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def take(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


class SolanaRPCClient:
    """Read-only async Solana JSON-RPC client.

    Use as an async context manager.
    """

    def __init__(
        self,
        rpc_url: str,
        *,
        http_timeout_seconds: float = 15.0,
        requests_per_second: float = 8.0,
    ) -> None:
        if not rpc_url:
            raise ValueError("rpc_url must be non-empty")
        self.rpc_url = rpc_url
        self._timeout = http_timeout_seconds
        self._bucket = TokenBucket(rate_per_second=requests_per_second)
        self._client: httpx.AsyncClient | None = None
        self._req_id = 0

    async def __aenter__(self) -> SolanaRPCClient:
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _call(self, method: str, params: list[Any]) -> Any:
        assert self._client is not None, "use async with"
        await self._bucket.take()
        self._req_id += 1
        body = {
            "jsonrpc": "2.0",
            "id": self._req_id,
            "method": method,
            "params": params,
        }

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
            retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
            reraise=True,
        ):
            with attempt:
                resp = await self._client.post(self.rpc_url, json=body)
                if resp.status_code in (429, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"transient {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                if resp.status_code >= 400:
                    raise RPCError(f"RPC {method} failed: {resp.status_code} {resp.text[:300]}")
                data = resp.json()
                if "error" in data:
                    raise RPCError(f"RPC {method} error: {data['error']}")
                return data.get("result")
        return None  # pragma: no cover

    # ── high-level helpers ─────────────────────────────────────────────────

    async def get_slot(self) -> int:
        result = await self._call("getSlot", [{"commitment": "confirmed"}])
        return int(result)

    async def get_block_time(self, slot: int) -> int | None:
        result = await self._call("getBlockTime", [slot])
        return int(result) if result is not None else None

    async def get_signatures_for_address(
        self,
        address: str,
        *,
        limit: int = 100,
        before: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        params: list[Any] = [
            address,
            {"limit": min(max(1, limit), 1000)},
        ]
        if before:
            params[1]["before"] = before
        if until:
            params[1]["until"] = until
        result = await self._call("getSignaturesForAddress", params)
        return list(result or [])

    async def get_transaction(
        self,
        signature: str,
        *,
        max_supported_transaction_version: int = 0,
    ) -> dict[str, Any] | None:
        result = await self._call(
            "getTransaction",
            [
                signature,
                {
                    "encoding": "jsonParsed",
                    "maxSupportedTransactionVersion": max_supported_transaction_version,
                    "commitment": "confirmed",
                },
            ],
        )
        return result if isinstance(result, dict) else None

    async def get_account_info(self, address: str) -> dict[str, Any] | None:
        result = await self._call(
            "getAccountInfo",
            [address, {"encoding": "base64", "commitment": "confirmed"}],
        )
        if not isinstance(result, dict):
            return None
        return result.get("value") if isinstance(result.get("value"), dict) else result

    async def get_token_supply(self, mint: str) -> dict[str, Any] | None:
        result = await self._call(
            "getTokenSupply",
            [mint, {"commitment": "confirmed"}],
        )
        if not isinstance(result, dict):
            return None
        return result.get("value") if isinstance(result.get("value"), dict) else result
