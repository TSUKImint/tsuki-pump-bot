"""Per-mint token state aggregator.

Consumes `PumpEvent`s from `PumpScout` and folds them into `TokenState`
objects. Maintains:

  - per-mint `TokenState` with dev wallet, buy log, curve metrics
  - 60s sliding window of distinct buyers
  - rolling SOL velocity (SOL/min added to curve)

LRU-bounded so memory stays flat over long runs.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

from ..models import BuyRecord, TokenState
from ..solana.bonding_curve import (
    DEFAULT_VIRTUAL_SOL_RESERVES,
    DEFAULT_VIRTUAL_TOKEN_RESERVES,
    LAMPORTS_PER_SOL,
    TOKEN_UNIT_MULTIPLIER,
    BondingCurveState,
    buy_cost_sol,
)
from ..solana.pump_program import PumpEvent, PumpInstructionKind

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class _SolPoint:
    ts_unix: int
    sol_added: float


class TokenStateAggregator:
    """Maintain TokenState across all tokens we've seen."""

    def __init__(self, *, max_tokens: int = 10_000) -> None:
        self.max_tokens = max_tokens
        self._tokens: OrderedDict[str, TokenState] = OrderedDict()
        self._sol_history: dict[str, deque[_SolPoint]] = {}

    def __len__(self) -> int:
        return len(self._tokens)

    def get(self, mint: str) -> TokenState | None:
        token = self._tokens.get(mint)
        if token is not None:
            self._tokens.move_to_end(mint)
        return token

    def all_tokens(self) -> list[TokenState]:
        return list(self._tokens.values())

    def _ensure(self, mint: str, *, dev_wallet: str | None = None) -> TokenState:
        token = self._tokens.get(mint)
        if token is None:
            token = TokenState(mint=mint, dev_wallet=dev_wallet)
            token.first_seen_at = datetime.now(tz=UTC)
            self._tokens[mint] = token
            self._sol_history[mint] = deque(maxlen=200)
            while len(self._tokens) > self.max_tokens:
                evicted_mint, _ = self._tokens.popitem(last=False)
                self._sol_history.pop(evicted_mint, None)
        else:
            self._tokens.move_to_end(mint)
            if dev_wallet and not token.dev_wallet:
                token.dev_wallet = dev_wallet
        return token

    def ingest(self, event: PumpEvent) -> TokenState | None:
        """Fold `event` into the matching `TokenState`. Returns the updated
        token (or None if we couldn't associate the event)."""
        mint = event.mint
        if not mint:
            return None

        if event.kind == PumpInstructionKind.CREATE:
            token = self._ensure(mint, dev_wallet=event.dev_wallet)
            token.created_at_unix = event.block_time_unix
            if event.dev_wallet:
                token.dev_activity.last_seen_unix = event.block_time_unix
            return token

        if event.kind == PumpInstructionKind.BUY:
            token = self._ensure(mint)
            buy = BuyRecord(
                wallet=event.actor_wallet or "",
                sol_spent=event.sol_amount or 0.0,
                token_units_received=event.token_amount or 0.0,
                slot=event.slot,
                block_time_unix=event.block_time_unix,
                signature=event.signature,
            )
            token.add_buy(buy)
            if event.sol_amount:
                token.sol_in_curve += event.sol_amount
                self._update_velocity(mint, token, event.sol_amount, event.block_time_unix)
                token.last_price_sol_per_token = self._estimate_price(token)
            self._update_distinct_buyers_60s(token, event)
            if event.dev_wallet and token.dev_activity.last_seen_unix is None:
                token.dev_activity.last_seen_unix = event.block_time_unix
            if token.dev_wallet and event.actor_wallet == token.dev_wallet:
                token.dev_activity.last_seen_unix = event.block_time_unix
            return token

        if event.kind == PumpInstructionKind.SELL:
            token = self._ensure(mint)
            if event.actor_wallet and event.actor_wallet == token.dev_wallet:
                token.dev_activity.last_seen_unix = event.block_time_unix
                if event.sol_amount:
                    token.dev_activity.cumulative_sol_pulled += event.sol_amount
            return token

        return None

    def _update_velocity(
        self,
        mint: str,
        token: TokenState,
        sol_added: float,
        ts_unix: int | None,
    ) -> None:
        if ts_unix is None:
            ts_unix = int(time.time())
        history = self._sol_history.setdefault(mint, deque(maxlen=200))
        history.append(_SolPoint(ts_unix=ts_unix, sol_added=sol_added))
        # Window: last 60 seconds inclusive. Sum SOL added in the window → that
        # is already SOL-per-minute (window length is 60 s by construction).
        cutoff = ts_unix - 60
        while history and history[0].ts_unix < cutoff:
            history.popleft()
        token.last_sol_velocity_sol_per_min = sum(p.sol_added for p in history)

    @staticmethod
    def _update_distinct_buyers_60s(token: TokenState, event: PumpEvent) -> None:
        if event.block_time_unix is None:
            return
        cutoff = event.block_time_unix - 60
        seen: set[str] = set()
        for buy in token.buys:
            if buy.block_time_unix is None:
                continue
            if buy.block_time_unix >= cutoff:
                seen.add(buy.wallet)
        token.distinct_buyers_60s = len(seen)

    @staticmethod
    def _estimate_price(token: TokenState) -> float:
        """Estimate spot price using the curve math assuming defaults.

        The default virtual reserves are correct for newly-created tokens.
        For older tokens we'd want to read the live BondingCurve account;
        we'll wire that in alongside the live executor in v0.3.
        """
        # Approximate: real_sol_reserves = sol_in_curve (lamports).
        real_sol_lamports = int(token.sol_in_curve * LAMPORTS_PER_SOL)
        state = BondingCurveState(
            virtual_sol_reserves=DEFAULT_VIRTUAL_SOL_RESERVES + real_sol_lamports,
            virtual_token_reserves=DEFAULT_VIRTUAL_TOKEN_RESERVES,
            real_sol_reserves=real_sol_lamports,
            real_token_reserves=0,
            complete=token.curve_complete,
        )
        # Price = cost to buy 1 token, with 6 decimals.
        return buy_cost_sol(state, 1.0 / TOKEN_UNIT_MULTIPLIER) * TOKEN_UNIT_MULTIPLIER
