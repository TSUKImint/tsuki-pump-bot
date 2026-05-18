"""KOL / whale discovery from on-chain history.

The `first_kol_touch` and `convergence` filters depend on a curated
leaderboard of profitable Solana wallets (CSV at
``data/private_kol_list.csv``). Out of the box that file is empty, so
35% of the composite score weight returns neutral.

This package automates the curation. Given a stream of historical
pump.fun events it:

1. Classifies the outcome of each token (graduated / 2x+ / dead / pending).
2. Builds per-wallet features (hit rate, log-ROI mean, recency, n_trades).
3. **Rejects unsafe candidates BEFORE scoring**:
   - **Bots** — wallets with high tx-frequency, slot-zero entries on many
     tokens, mechanically repetitive trade sizes / hold-times.
   - **Poison wallets** — wallets whose wins all stem from tokens funded
     by the same upstream wallet (dev-funded honey wallets), or that
     consistently exit to the same destination (a "dump bus" address).
   - **Abandoned wallets** — must have recent activity (configurable
     window) and a minimum number of recent trades.
   - **Survivorship-biased winners** — wallets whose ROI is dominated
     by 1-2 lucky multi-bagger trades (use geometric / log-ROI mean,
     not arithmetic, so a single 100x can't single-handedly anoint a
     wallet).

The output is a sorted ``KolEntry`` list that can be written straight
to ``data/private_kol_list.csv`` for the existing filters to consume —
no filter-side changes required.

This module is **deliberately I/O-free**. Callers supply iterables of
``PumpEvent`` (from the existing scout / a recorded firehose / the
event store) plus optional lookup tables for the funder graph; the
discovery module only does the scoring. That keeps it pure-functional,
testable, and reusable across data sources.
"""

from .bot_heuristics import BotHeuristicsConfig, is_likely_bot
from .kol_discovery import (
    DiscoveredKol,
    KolDiscoveryConfig,
    discover_kols,
)
from .outcomes import (
    TokenOutcome,
    TokenOutcomeLabel,
    classify_token_outcomes,
)
from .poison_heuristics import PoisonHeuristicsConfig, is_likely_poison_wallet
from .wallet_features import WalletFeatures, build_wallet_features

__all__ = [
    "BotHeuristicsConfig",
    "DiscoveredKol",
    "KolDiscoveryConfig",
    "PoisonHeuristicsConfig",
    "TokenOutcome",
    "TokenOutcomeLabel",
    "WalletFeatures",
    "build_wallet_features",
    "classify_token_outcomes",
    "discover_kols",
    "is_likely_bot",
    "is_likely_poison_wallet",
]
