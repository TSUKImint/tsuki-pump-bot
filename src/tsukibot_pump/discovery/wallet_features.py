"""Per-wallet feature extraction.

Walks the event stream once per wallet and computes the features the
scoring stage needs:

* ``hit_rate`` — fraction of early buys on tokens whose outcome was
  ``WINNER`` or ``GRADUATED``. Trades on PENDING tokens (still inside the
  observation window) are excluded so we don't penalise wallets for
  recent trades that haven't had time to play out.
* ``log_roi_mean`` — geometric mean of (peak / first-price) for each
  positive-outcome trade. ``log`` to protect against survivorship: one
  100x can't single-handedly anoint a wallet because we average log
  multiples, not raw multiples.
* ``trades_per_day`` — total observed buys / window length, used for
  bot detection.
* ``recency`` — seconds since the wallet's most recent observed buy.
* ``same_slot_entry_fraction`` — fraction of buys placed in slot
  ``token_create_slot + [0..N]``, used for sniper-bot detection.
* ``sol_size_cv`` — coefficient of variation of SOL trade sizes.
* ``hold_time_seconds_cv`` — coefficient of variation of buy→sell
  hold times for round-trip trades on the same mint.

We compute features for **every** wallet observed buying any pump.fun
token. Whether they're considered as KOLs is decided downstream by the
``discover_kols`` orchestrator — features are pure, decisions are
configurable.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..solana.pump_program import PumpEvent, PumpInstructionKind
from .bot_heuristics import coefficient_of_variation

if TYPE_CHECKING:
    from .outcomes import TokenOutcome


@dataclass(slots=True)
class WalletFeatures:
    wallet: str
    n_observed_buys: int = 0
    n_observed_sells: int = 0
    n_observed_round_trips: int = 0
    n_distinct_mints: int = 0
    first_seen_unix: int | None = None
    last_seen_unix: int | None = None
    # Outcome-conditional counts (computed against a TokenOutcome map).
    n_trades_on_winners: int = 0
    n_trades_on_dead: int = 0
    n_trades_on_pending: int = 0
    log_roi_sum: float = 0.0
    log_roi_count: int = 0
    # Bot-detection features.
    same_slot_entries: int = 0
    sol_amounts: list[float] = field(default_factory=list)
    hold_times_seconds: list[float] = field(default_factory=list)
    # Internal: for hit_rate denominator we exclude pending trades.
    _decided_trades: int = 0

    # ── derived metrics ──────────────────────────────────────────────────

    @property
    def hit_rate(self) -> float:
        if self._decided_trades == 0:
            return 0.0
        return self.n_trades_on_winners / self._decided_trades

    @property
    def log_roi_mean(self) -> float:
        if self.log_roi_count == 0:
            return 0.0
        return self.log_roi_sum / self.log_roi_count

    @property
    def trades_per_day(self) -> float:
        if self.first_seen_unix is None or self.last_seen_unix is None:
            return 0.0
        span_seconds = max(1, self.last_seen_unix - self.first_seen_unix)
        return self.n_observed_buys / (span_seconds / 86_400.0)

    @property
    def same_slot_entry_fraction(self) -> float:
        if self.n_observed_buys == 0:
            return 0.0
        return self.same_slot_entries / self.n_observed_buys

    @property
    def sol_size_cv(self) -> float:
        return coefficient_of_variation(self.sol_amounts)

    @property
    def hold_time_seconds_cv(self) -> float | None:
        if len(self.hold_times_seconds) < 2:
            return None
        return coefficient_of_variation(self.hold_times_seconds)


def build_wallet_features(
    events: Iterable[PumpEvent],
    outcomes: Mapping[str, TokenOutcome],
    *,
    create_slot_proximity_window: int = 3,
) -> dict[str, WalletFeatures]:
    """Walk events once, returning a per-wallet feature map.

    Pure function over (events, outcomes). The same input yields the
    same output regardless of when it's called.
    """
    feats: dict[str, WalletFeatures] = {}

    # Track each mint's create slot (for slot-proximity detection) and
    # each wallet's open buy slot per mint (for hold-time computation).
    create_slot: dict[str, int] = {}
    open_buy_slot_unix: dict[tuple[str, str], int] = defaultdict(int)  # (mint, wallet) -> ts
    mint_seen_by_wallet: dict[str, set[str]] = defaultdict(set)

    for ev in events:
        if ev.mint and ev.kind == PumpInstructionKind.CREATE:
            create_slot[ev.mint] = ev.slot
            continue

        if ev.actor_wallet is None or ev.mint is None:
            continue

        wallet = ev.actor_wallet
        feat = feats.setdefault(wallet, WalletFeatures(wallet=wallet))
        ts = ev.block_time_unix

        if ts is not None:
            if feat.first_seen_unix is None or ts < feat.first_seen_unix:
                feat.first_seen_unix = ts
            if feat.last_seen_unix is None or ts > feat.last_seen_unix:
                feat.last_seen_unix = ts

        if ev.kind == PumpInstructionKind.BUY:
            feat.n_observed_buys += 1
            mint_seen_by_wallet[wallet].add(ev.mint)
            sol = float(ev.sol_amount or 0.0)
            if sol > 0:
                feat.sol_amounts.append(sol)

            # Slot proximity → sniper signal.
            ms = create_slot.get(ev.mint)
            if ms is not None and ev.slot <= ms + create_slot_proximity_window:
                feat.same_slot_entries += 1

            # Open buy slot for hold-time calculation (track first buy on
            # each mint to avoid bias from re-buys).
            key = (ev.mint, wallet)
            if key not in open_buy_slot_unix and ts is not None:
                open_buy_slot_unix[key] = ts

            # Outcome-conditional tallies.
            outcome = outcomes.get(ev.mint)
            if outcome is not None:
                label = outcome.label.value
                if label in ("winner", "graduated"):
                    feat.n_trades_on_winners += 1
                    feat._decided_trades += 1
                    if outcome.peak_multiple > 1.0:
                        feat.log_roi_sum += math.log(outcome.peak_multiple)
                        feat.log_roi_count += 1
                elif label in ("neutral", "dead"):
                    feat.n_trades_on_dead += 1
                    feat._decided_trades += 1
                else:  # pending
                    feat.n_trades_on_pending += 1

        elif ev.kind == PumpInstructionKind.SELL:
            feat.n_observed_sells += 1
            key = (ev.mint, wallet)
            if key in open_buy_slot_unix and ts is not None:
                hold_seconds = ts - open_buy_slot_unix.pop(key)
                if hold_seconds > 0:
                    feat.hold_times_seconds.append(float(hold_seconds))
                    feat.n_observed_round_trips += 1

    for wallet, feat in feats.items():
        feat.n_distinct_mints = len(mint_seen_by_wallet[wallet])
    return feats
