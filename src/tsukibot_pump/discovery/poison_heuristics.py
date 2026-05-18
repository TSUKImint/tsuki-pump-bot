"""Poison-wallet detection.

A "poison wallet" is an honest-looking trader account that is secretly
controlled by a token launcher or a coordinated group, designed to bait
copy-traders into following them.

Pattern we're defending against:

1. Dev launches token T1. Dev funds wallet W with SOL from a sibling
   wallet (or via Jito tipping infrastructure). W "buys early" on T1.
2. Dev pumps T1 just enough to print a 5x for W, then dumps.
3. Copy-traders see W's 5x and start following.
4. Dev launches T2. W "buys early" on T2 using SOL from the same
   upstream wallet. Copy-traders pile in. Dev dumps on the copy-traders.

The wins on W are real, but they cluster around tokens funded by a
common upstream wallet. That's the tell.

We detect this by looking at the **funder graph**: every wallet's
ultimate SOL source. If too high a fraction of a wallet's winning trades
are on tokens whose dev wallets share a common funder (or are the same
funder as the wallet itself), we treat the wallet as poisoned and drop
it from the leaderboard.

We also flag wallets whose sells consistently route to the **same
destination address**. A real KOL's profits go to a personal wallet
that varies over time (CEX deposit, multisig, cold storage rotation).
A poison wallet's profits drain to a fixed dump-bus address.

Both checks require lookup tables the caller must populate (we
deliberately don't make `getSignaturesForAddress` calls here — that's
I/O and we want this module pure). The lookups are:

* ``funder_lookup[wallet] -> upstream_funder_wallet`` — the first
  inbound transfer source for ``wallet``. Often the same across multiple
  related accounts.
* ``sell_destination_lookup[wallet] -> list[destination_wallet]`` —
  every destination this wallet has ever sent SOL to. Useful for
  spotting fixed dump-bus addresses.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from .outcomes import TokenOutcome


@dataclass(frozen=True, slots=True)
class PoisonHeuristicsConfig:
    # Reject if >= this fraction of the wallet's winning trades are on
    # tokens whose dev wallets share a common upstream funder.
    max_shared_funder_fraction_of_wins: float = 0.50
    # Reject if the wallet's own funder is the dev of any of its winning
    # trades' tokens. That's the smoking gun — the "KOL" is literally the
    # dev's funded sibling wallet.
    reject_if_own_funder_is_dev_of_winner: bool = True
    # Reject if >= this fraction of the wallet's sell destinations go to a
    # single address (a "dump bus").
    max_dump_bus_destination_fraction: float = 0.70
    # Skip these checks below this many positive trades — not enough data
    # to draw the cluster.
    min_positive_trades_for_check: int = 3


def is_likely_poison_wallet(
    wallet: str,
    positive_outcomes: list[TokenOutcome],
    *,
    config: PoisonHeuristicsConfig,
    funder_lookup: Mapping[str, str] | None = None,
    sell_destination_lookup: Mapping[str, list[str]] | None = None,
) -> tuple[bool, str]:
    """Return ``(flagged, reason)``. Empty reason iff not flagged.

    ``positive_outcomes`` is the subset of token outcomes (winner or
    graduated) where this wallet was an early buyer.
    """
    if len(positive_outcomes) < config.min_positive_trades_for_check:
        return False, ""

    funder_lookup = funder_lookup or {}
    sell_destination_lookup = sell_destination_lookup or {}

    # 1) Own funder == dev of any winner.
    own_funder = funder_lookup.get(wallet)
    if config.reject_if_own_funder_is_dev_of_winner and own_funder:
        for outcome in positive_outcomes:
            if outcome.dev_wallet and outcome.dev_wallet == own_funder:
                return True, (
                    f"wallet's funder {own_funder[:8]} is also the dev of "
                    f"winning token {outcome.mint[:8]} — likely dev-funded honey"
                )

    # 2) Common funder shared across many wins.
    dev_funders = [
        funder_lookup.get(outcome.dev_wallet) for outcome in positive_outcomes if outcome.dev_wallet
    ]
    dev_funders_filtered = [f for f in dev_funders if f]
    if dev_funders_filtered:
        counts = Counter(dev_funders_filtered)
        top_funder, top_count = counts.most_common(1)[0]
        share = top_count / max(1, len(positive_outcomes))
        if share >= config.max_shared_funder_fraction_of_wins:
            return True, (
                f"{share:.0%} of wins funded by upstream wallet "
                f"{top_funder[:8]} — likely coordinated"
            )

    # 3) Dump-bus destination concentration.
    destinations = sell_destination_lookup.get(wallet, [])
    if destinations:
        dest_counts = Counter(destinations)
        top_dest, top_dest_count = dest_counts.most_common(1)[0]
        dest_share = top_dest_count / len(destinations)
        if dest_share >= config.max_dump_bus_destination_fraction:
            return True, (
                f"{dest_share:.0%} of sells routed to {top_dest[:8]} — likely dump-bus address"
            )

    return False, ""
