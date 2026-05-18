"""Per-token outcome classification.

A "winner" is a token whose price peaked at >= ``min_multiple`` x the first
recorded fill price within ``observation_window_seconds`` after launch. A
"graduated" token additionally crossed the bonding-curve graduation
threshold (~85 SOL of real reserves). A "dead" token's peak was <2x.

We deliberately use peak-price multiples (not realized P&L) because:

* The KOL we want to identify is the one who *bought early* and *had the
  option* to sell at the peak. Whether they actually sold is irrelevant
  to the question "was this a good token to be early on?".
* Realized P&L per wallet conflates good selection with good exit
  timing. Selection is what we're scoring here; exit timing is a
  separate problem the position monitor handles.

We could also tag tokens whose price dropped >90% from peak within 24h
as "rugs" and treat early-buy presence on those as a small negative
signal. That's left as a future enhancement — we don't want to punish
KOLs who routinely take 2x and run before the rug.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from ..solana.pump_program import PumpEvent, PumpInstructionKind


class TokenOutcomeLabel(StrEnum):
    """High-level outcome of a token over the observation window."""

    GRADUATED = "graduated"
    WINNER = "winner"  # peak >= min_multiple_winner * first_price
    NEUTRAL = "neutral"  # peak >= 1.2x but < min_multiple_winner
    DEAD = "dead"  # peak < 1.2x
    PENDING = "pending"  # observation window not complete


@dataclass(frozen=True, slots=True)
class TokenOutcome:
    mint: str
    dev_wallet: str | None
    first_seen_unix: int | None
    first_price_sol_per_token: float
    peak_price_sol_per_token: float
    peak_multiple: float
    last_real_sol_in_curve: float
    label: TokenOutcomeLabel
    # Buyers in chronological order, used by feature extraction to compute
    # "was this wallet among the first N buyers" without re-scanning events.
    early_buyer_wallets: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class _TokenAccumulator:
    mint: str
    dev_wallet: str | None = None
    first_seen_unix: int | None = None
    first_price: float = 0.0
    peak_price: float = 0.0
    real_sol_in_curve: float = 0.0
    buyers_in_order: list[str] = field(default_factory=list)
    seen_buyers: set[str] = field(default_factory=set)


def classify_token_outcomes(
    events: Iterable[PumpEvent],
    *,
    min_multiple_winner: float = 3.0,
    graduation_real_sol_threshold: float = 85.0,
    observation_window_seconds: int = 24 * 3600,
    now_unix: int | None = None,
    early_buyer_count: int = 50,
) -> dict[str, TokenOutcome]:
    """Walk events once, returning a per-mint outcome map.

    Pure function: produces the same result for the same input regardless
    of when it's called. ``now_unix`` is only used to mark observations
    as ``PENDING`` (still within their window) — leaving it ``None``
    classifies everything based purely on observed events.
    """
    accs: dict[str, _TokenAccumulator] = defaultdict(lambda: _TokenAccumulator(mint=""))

    for ev in events:
        if not ev.mint:
            continue
        acc = accs.setdefault(ev.mint, _TokenAccumulator(mint=ev.mint))
        if acc.first_seen_unix is None and ev.block_time_unix is not None:
            acc.first_seen_unix = ev.block_time_unix
        if ev.dev_wallet and not acc.dev_wallet:
            acc.dev_wallet = ev.dev_wallet

        if ev.kind == PumpInstructionKind.BUY:
            if ev.sol_amount:
                acc.real_sol_in_curve += float(ev.sol_amount)
            if ev.actor_wallet and ev.actor_wallet not in acc.seen_buyers:
                acc.seen_buyers.add(ev.actor_wallet)
                acc.buyers_in_order.append(ev.actor_wallet)

            # Approximate instantaneous price from cumulative SOL & token
            # amounts if both are present; otherwise treat the SOL-amount /
            # token-amount ratio of the individual buy as a price sample.
            price = _approx_price_from_buy(ev)
            if price > 0:
                if acc.first_price == 0:
                    acc.first_price = price
                if price > acc.peak_price:
                    acc.peak_price = price

        elif ev.kind == PumpInstructionKind.SELL:
            if ev.sol_amount:
                acc.real_sol_in_curve = max(0.0, acc.real_sol_in_curve - float(ev.sol_amount))

    out: dict[str, TokenOutcome] = {}
    for mint, acc in accs.items():
        first_price = acc.first_price
        peak_price = max(acc.peak_price, first_price)
        peak_multiple = peak_price / first_price if first_price > 0 else 0.0

        label = _label(
            acc=acc,
            first_price=first_price,
            peak_multiple=peak_multiple,
            now_unix=now_unix,
            observation_window_seconds=observation_window_seconds,
            min_multiple_winner=min_multiple_winner,
            graduation_real_sol_threshold=graduation_real_sol_threshold,
        )
        out[mint] = TokenOutcome(
            mint=mint,
            dev_wallet=acc.dev_wallet,
            first_seen_unix=acc.first_seen_unix,
            first_price_sol_per_token=first_price,
            peak_price_sol_per_token=peak_price,
            peak_multiple=peak_multiple,
            last_real_sol_in_curve=acc.real_sol_in_curve,
            label=label,
            early_buyer_wallets=tuple(acc.buyers_in_order[:early_buyer_count]),
        )
    return out


def _approx_price_from_buy(ev: PumpEvent) -> float:
    sol = float(ev.sol_amount or 0.0)
    tok = float(ev.token_amount or 0.0)
    if sol <= 0 or tok <= 0:
        return 0.0
    return sol / tok


def _label(
    *,
    acc: _TokenAccumulator,
    first_price: float,
    peak_multiple: float,
    now_unix: int | None,
    observation_window_seconds: int,
    min_multiple_winner: float,
    graduation_real_sol_threshold: float,
) -> TokenOutcomeLabel:
    if first_price <= 0:
        # We didn't see enough to estimate price — treat as pending so
        # nothing gets scored on it.
        return TokenOutcomeLabel.PENDING

    # If now_unix is provided, mark tokens still inside their window as
    # PENDING — don't use partial data to call a winner.
    if (
        now_unix is not None
        and acc.first_seen_unix is not None
        and (now_unix - acc.first_seen_unix) < observation_window_seconds
    ):
        return TokenOutcomeLabel.PENDING

    if acc.real_sol_in_curve >= graduation_real_sol_threshold:
        return TokenOutcomeLabel.GRADUATED
    if peak_multiple >= min_multiple_winner:
        return TokenOutcomeLabel.WINNER
    if peak_multiple >= 1.2:
        return TokenOutcomeLabel.NEUTRAL
    return TokenOutcomeLabel.DEAD


def is_positive_outcome(label: TokenOutcomeLabel) -> bool:
    return label in (TokenOutcomeLabel.GRADUATED, TokenOutcomeLabel.WINNER)
