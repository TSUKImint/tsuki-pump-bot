"""Aggressive paper-trader profile + early-conviction lane logic.

Both pieces are opt-in (controlled by `scoring.aggressive_paper.enabled`
and `scoring.early_conviction.enabled` respectively) and have no effect
when off — v0.2 configs keep behaving the way they used to.

`apply_aggressive_overrides` rewrites a `Config` in-place so the rest of
the orchestrator can stay unchanged. It returns the (possibly modified)
config along with an `EffectiveProfile` summary that the bot logs at
startup so the user can see exactly what was changed.

The early-conviction lane is implemented as a pure predicate
(`is_early_conviction_signal`) plus a constraint clamp on the size
(`clamp_for_early_conviction`); the orchestrator wires the predicate into
its entry decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .models import TokenState


@dataclass(frozen=True, slots=True)
class EffectiveProfile:
    """What `apply_aggressive_overrides` actually changed.

    Logged once at startup so the user can audit it.
    """

    aggressive_enabled: bool
    enter_threshold: float
    fraction_of_kelly: float
    min_expected_roi: float
    single_token_cap_fraction: float
    enter_after_sol_in_curve_gte: float
    min_velocity_sol_per_min: float
    http_poll_interval_seconds: float
    scoring_cycle_seconds: float
    early_conviction_enabled: bool


def apply_aggressive_overrides(config: Config) -> tuple[Config, EffectiveProfile]:
    """Apply the aggressive-paper profile to `config`.

    Returns a (mutated_config, EffectiveProfile) tuple. When the profile is
    disabled, returns `config` unchanged and a profile snapshot of the
    *current* values (so callers can log them uniformly).
    """
    prof = config.scoring.aggressive_paper
    ec = config.scoring.early_conviction

    if not prof.enabled:
        return config, EffectiveProfile(
            aggressive_enabled=False,
            enter_threshold=config.scoring.enter_threshold,
            fraction_of_kelly=config.sizing.fraction_of_kelly,
            min_expected_roi=config.sizing.min_expected_roi,
            single_token_cap_fraction=config.bankroll.single_token_cap_fraction,
            enter_after_sol_in_curve_gte=config.filters.curve_graduation.enter_after_sol_in_curve_gte,
            min_velocity_sol_per_min=config.filters.curve_graduation.min_velocity_sol_per_min,
            http_poll_interval_seconds=config.watch.http_poll_interval_seconds,
            scoring_cycle_seconds=5.0,  # the default hard-coded in __main__
            early_conviction_enabled=ec.enabled,
        )

    # Pydantic v2: `model_copy(update=...)` returns a new model. We then
    # reassemble `Config` from those new sub-models; this preserves
    # validation (since we go through the same schema).
    new_scoring = config.scoring.model_copy(update={"enter_threshold": prof.enter_threshold})
    new_sizing = config.sizing.model_copy(
        update={
            "fraction_of_kelly": prof.fraction_of_kelly,
            "min_expected_roi": prof.min_expected_roi,
        }
    )
    new_bankroll = config.bankroll.model_copy(
        update={"single_token_cap_fraction": prof.single_token_cap_fraction}
    )
    new_curve = config.filters.curve_graduation.model_copy(
        update={
            "enter_after_sol_in_curve_gte": prof.enter_after_sol_in_curve_gte,
            "min_velocity_sol_per_min": prof.min_velocity_sol_per_min,
        }
    )
    new_filters = config.filters.model_copy(update={"curve_graduation": new_curve})
    new_watch = config.watch.model_copy(
        update={"http_poll_interval_seconds": prof.http_poll_interval_seconds}
    )

    new_config = config.model_copy(
        update={
            "scoring": new_scoring,
            "sizing": new_sizing,
            "bankroll": new_bankroll,
            "filters": new_filters,
            "watch": new_watch,
        }
    )

    return new_config, EffectiveProfile(
        aggressive_enabled=True,
        enter_threshold=prof.enter_threshold,
        fraction_of_kelly=prof.fraction_of_kelly,
        min_expected_roi=prof.min_expected_roi,
        single_token_cap_fraction=prof.single_token_cap_fraction,
        enter_after_sol_in_curve_gte=prof.enter_after_sol_in_curve_gte,
        min_velocity_sol_per_min=prof.min_velocity_sol_per_min,
        http_poll_interval_seconds=prof.http_poll_interval_seconds,
        scoring_cycle_seconds=prof.scoring_cycle_seconds,
        early_conviction_enabled=ec.enabled,
    )


# ── Early-conviction lane ─────────────────────────────────────────────────


def count_kol_touches_in_window(
    token: TokenState, kol_wallets: set[str], window_seconds: int
) -> int:
    """Count distinct KOL wallets that bought within `window_seconds` of CREATE.

    The token must have `created_at_unix` set. Returns 0 if not.
    """
    if token.created_at_unix is None or not kol_wallets:
        return 0
    cutoff = token.created_at_unix + window_seconds
    seen: set[str] = set()
    for buy in token.buys:
        if not buy.wallet:
            continue
        if buy.block_time_unix is None:
            continue
        if buy.block_time_unix > cutoff:
            continue
        if buy.wallet in kol_wallets:
            seen.add(buy.wallet)
    return len(seen)


def is_early_conviction_signal(
    token: TokenState,
    *,
    kol_wallets: set[str],
    window_seconds: int,
    min_kol_touches: int,
    min_prior_graduations: int,
) -> bool:
    """Return True iff the token meets all early-conviction-lane gates.

    Gates (all required):
      1. token has a known on-chain `creator` field
      2. creator has >= `min_prior_graduations` prior graduations
      3. >= `min_kol_touches` distinct tracked KOLs bought within
         `window_seconds` of CREATE
    """
    if not token.creator:
        return False
    if token.creator_prior_graduations < min_prior_graduations:
        return False
    touches = count_kol_touches_in_window(token, kol_wallets, window_seconds)
    return touches >= min_kol_touches


def early_conviction_cap_sol(
    config: Config,
    bankroll_sol: float,
) -> float:
    """Hard SOL cap for an early-conviction entry.

    Bounded above by both (a) the user's normal single-token cap and
    (b) the early-conviction lane's own `max_single_token_cap_fraction`.
    """
    normal_cap = config.bankroll.total_sol * config.bankroll.single_token_cap_fraction
    lane_cap = bankroll_sol * config.scoring.early_conviction.max_single_token_cap_fraction
    return max(0.0, min(normal_cap, lane_cap))
