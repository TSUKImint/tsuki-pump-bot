"""tsuki-pump-bot — pump.fun watchtower + paper-trader.

Selection-edge bot: refuses to buy 99% of new tokens, fires on the rare
survivors that pass a 6-filter stack (dev-wallet reputation, bundle/cluster
detection, first-KOL-touch, KOL convergence, bonding-curve graduation
prediction, community-takeover detection).

Paper-mock + paper modes default. Devnet / mainnet executors are gated
behind explicit double-confirmation.
"""

__version__ = "0.2.0"
