"""Shared in-memory data models.

Kept in a single module so filters / scorer / executor share the same view
of a token and can be tested independently of the live firehose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class BuyRecord:
    """A single buy observed against a bonding curve."""

    wallet: str
    sol_spent: float
    token_units_received: float
    slot: int
    block_time_unix: int | None
    signature: str


@dataclass(slots=True)
class DevActivity:
    """Tracked dev (creator) wallet behavior post-launch."""

    last_seen_unix: int | None = None
    sol_balance_at_launch: float | None = None
    token_balance_at_launch: float | None = None
    cumulative_sol_pulled: float = 0.0
    cumulative_tokens_sold: float = 0.0


@dataclass(slots=True)
class TokenState:
    """All the information the filter stack needs about a single token.

    Populated incrementally by `PumpScout` as it consumes the firehose, then
    inspected (read-only) by filters and the composite scorer.
    """

    mint: str
    dev_wallet: str | None = None
    symbol: str | None = None
    name: str | None = None
    created_at_unix: int | None = None
    first_seen_at: datetime | None = None

    # Bonding curve tracking
    sol_in_curve: float = 0.0
    last_sol_velocity_sol_per_min: float = 0.0
    curve_complete: bool = False
    last_price_sol_per_token: float = 0.0
    # Optional on-chain bonding curve address (set by aggregator when known).
    bonding_curve_address: str | None = None

    # v0.3: from the BondingCurve.creator field (May 2025 protocol upgrade).
    # Distinct from dev_wallet (the transaction signer of CREATE) — usually
    # but not always the same pubkey. Used by the creator-vault filter.
    creator: str | None = None
    creator_vault_sol: float = 0.0
    creator_prior_graduations: int = 0
    creator_tokens_7d: int = 0

    # Buyer behavior
    buys: list[BuyRecord] = field(default_factory=list)
    distinct_buyers_60s: int = 0

    # Dev tracking
    dev_activity: DevActivity = field(default_factory=DevActivity)

    # Last-known scores (populated by the scorer for the dashboard).
    last_filter_scores: dict[str, float] = field(default_factory=dict)
    last_composite_score: float = 0.0
    last_decision_reason: str = ""
    rejected: bool = False
    reject_reason: str = ""

    # Free-form bag for filters to stash intermediate state.
    extra: dict[str, Any] = field(default_factory=dict)

    def add_buy(self, buy: BuyRecord) -> None:
        self.buys.append(buy)


@dataclass(slots=True)
class FilterOutcome:
    """Result of running a single filter on a token."""

    name: str
    score: float  # 0-100
    hard_reject: bool = False
    reject_reason: str = ""
    notes: str = ""  # short human-readable explanation


@dataclass(slots=True)
class CompositeScore:
    """Combined filter scores → entry decision."""

    score: float  # 0-100
    breakdown: dict[str, float] = field(default_factory=dict)
    hard_rejected: bool = False
    reject_reason: str = ""
    enter: bool = False
