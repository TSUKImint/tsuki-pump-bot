"""Bot / sniper detection rules.

A wallet is "bot-like" if its trading pattern looks mechanical. We don't
try to distinguish good bots from bad bots — we just exclude them
wholesale from KOL candidates because:

1. They're typically running on infrastructure we can't out-compete on
   speed. Copy-trading them is structurally worse than copy-trading a
   discretionary human KOL who buys after the same signal we could see.
2. Many "winning" wallets on pump.fun are dev-controlled sniper bots that
   are buying the dev's own launches. Tracking them is the same as
   copy-trading the dev, which is the opposite of edge.
3. Bot trading patterns are *unstable* — they flip strategies or get
   replaced overnight. A bot that was profitable last month tells you
   nothing about next month.

Signals we use:

* **Same-slot-as-create entries**: human KOLs can't physically enter the
  same slot a token is created in. If a wallet repeatedly enters tokens
  within the first ~3 slots, that's a sniper bot.
* **Mechanical trade size**: human KOLs vary trade size by conviction.
  Bots often fix the SOL amount (e.g., always 0.1 SOL). Low coefficient
  of variation of SOL amount → bot-like.
* **High trade frequency**: discretionary humans place maybe 10-50 trades
  a day at the absolute extreme. Bots place hundreds.
* **Hold-time uniformity**: bots that buy-and-flip on a fixed schedule
  show very tight hold-time variance.

All thresholds are configurable. Defaults are deliberately conservative
(prefer false negatives over false positives — better to miss a real
KOL than to anoint a dev-controlled bot).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .wallet_features import WalletFeatures


@dataclass(frozen=True, slots=True)
class BotHeuristicsConfig:
    # Trades whose slot is <= (create_slot + slot_proximity_window_slots)
    # are flagged as "same-slot as creation".
    slot_proximity_window_slots: int = 3
    # If a wallet's same-slot entries make up >= this fraction of its
    # observed buys, it's flagged as a sniper bot.
    max_same_slot_entry_fraction: float = 0.30
    # Max trades-per-day allowed for a candidate KOL. Above this we assume
    # the wallet is bot-driven.
    max_trades_per_day: float = 80.0
    # Minimum coefficient of variation of SOL trade size. Below this,
    # the wallet uses mechanically uniform sizes.
    min_sol_size_cv: float = 0.10
    # Minimum coefficient of variation of hold-time (in seconds) between
    # buy → sell of the same mint. Below this, the wallet uses a
    # mechanical exit timer.
    min_hold_time_cv: float = 0.10
    # If we observed fewer than this many trades for the wallet, we skip
    # CV checks (not enough samples). The wallet still has to pass the
    # other rules to clear bot detection.
    min_samples_for_cv: int = 10


def is_likely_bot(features: WalletFeatures, config: BotHeuristicsConfig) -> tuple[bool, str]:
    """Return ``(flagged, reason)``. Empty reason iff not flagged.

    Order matters: cheapest-first so we short-circuit on the strongest
    signals (slot-zero entries are dispositive).
    """
    if features.same_slot_entry_fraction >= config.max_same_slot_entry_fraction:
        return True, (
            f"same-slot-as-create entries = "
            f"{features.same_slot_entry_fraction:.0%} >= "
            f"{config.max_same_slot_entry_fraction:.0%}"
        )

    if features.trades_per_day > config.max_trades_per_day:
        return True, (
            f"trade frequency = {features.trades_per_day:.1f}/day > "
            f"{config.max_trades_per_day:.1f}/day"
        )

    if (
        features.n_observed_buys >= config.min_samples_for_cv
        and features.sol_size_cv < config.min_sol_size_cv
    ):
        return True, (
            f"mechanical trade-size (CV={features.sol_size_cv:.3f} < {config.min_sol_size_cv:.3f})"
        )

    if (
        features.n_observed_round_trips >= config.min_samples_for_cv
        and features.hold_time_seconds_cv is not None
        and features.hold_time_seconds_cv < config.min_hold_time_cv
    ):
        return True, (
            f"mechanical hold-time (CV={features.hold_time_seconds_cv:.3f} < "
            f"{config.min_hold_time_cv:.3f})"
        )

    return False, ""


def coefficient_of_variation(values: list[float]) -> float:
    """CV = stdev / mean. Returns ``inf`` for empty input, 0.0 for constant.

    Used by the wallet-feature builder for size and hold-time signals.
    Exposed at module level for unit-testability.
    """
    if not values:
        return float("inf")
    mean = statistics.fmean(values)
    if mean == 0:
        return 0.0
    if len(values) < 2:
        return 0.0
    stdev = statistics.pstdev(values)
    return stdev / abs(mean)
