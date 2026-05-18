"""Position sizing for directional memecoin bets.

Combinatorial arb sizing (tsuki-edge-bot/core/kelly.py) doesn't apply here:
pump.fun trades are *directional* — we believe the token will go up, but we
can be wrong. Classical fractional Kelly applies.

We feed Kelly an *expected* win probability and payoff, both derived from
the composite score. Score-to-edge mapping is deliberately conservative:
even a perfect-score token only gets sized to a quarter-Kelly equivalent
by default, because the underlying signal quality is unknown until we
have months of forward-test data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SizingInputs:
    """Inputs to the sizing function.

    `composite_score` is a 0-100 score from the composite scorer. We map it
    to a win probability assuming a roughly linear relationship from 50 (the
    "no edge" threshold) to 100 (max conviction). At score=50, win
    probability = 50% (no edge). At score=100, win probability = ~62%
    (mild edge — we deliberately stay humble on a market where 87% of tokens
    lose 90%+).

    Payoff (b in Kelly's bp - q / b) is set to 3.0: a winner is assumed to
    return 3x the loss size in expectation, based on the exits ladder
    (50% out at +100%, 30% at +300%, 20% rides with trailing stop).
    """

    bankroll_sol: float
    composite_score: float  # 0-100
    cost_per_unit_sol: float  # entry price per token unit
    fraction_of_kelly: float  # global throttle
    hard_cap_per_trade_sol: float
    single_token_cap_sol: float


@dataclass(frozen=True, slots=True)
class SizingResult:
    units: float
    notional_sol: float
    kelly_fraction_raw: float  # un-clamped Kelly fraction
    binding_constraint: str  # "kelly" | "capital" | "hard_cap"


def _score_to_probability(score: float) -> float:
    """Map composite score in [0, 100] to win probability in [0.5, 0.62].

    At score < 50, the bot shouldn't be entering anyway (gate is higher),
    so we cap below at 0.5 (no edge). Above 50 we add up to 12 percentage
    points based on score — deliberately humble because backtests will lie
    until we have a multi-month forward test.
    """
    if score <= 50:
        return 0.5
    return min(0.62, 0.5 + 0.12 * (score - 50) / 50)


def size_memecoin_position(inputs: SizingInputs) -> SizingResult:
    """Compute units to buy given score, bankroll, and risk caps.

    Returns a SizingResult. `units` can be 0 if Kelly returns a non-positive
    edge or if any cap is binding to zero.
    """
    if inputs.bankroll_sol <= 0:
        raise ValueError("bankroll_sol must be positive")
    if inputs.cost_per_unit_sol <= 0:
        raise ValueError("cost_per_unit_sol must be positive")
    if not 0 < inputs.fraction_of_kelly <= 1:
        raise ValueError("fraction_of_kelly must be in (0, 1]")

    p = _score_to_probability(inputs.composite_score)
    q = 1.0 - p
    b = 3.0  # assumed payoff ratio (see docstring)
    # Kelly: f* = (bp - q) / b
    kelly_raw = (b * p - q) / b
    kelly_throttled = max(0.0, kelly_raw * inputs.fraction_of_kelly)

    kelly_notional = inputs.bankroll_sol * kelly_throttled
    notional = min(
        kelly_notional,
        inputs.single_token_cap_sol,
        inputs.hard_cap_per_trade_sol,
    )

    if notional <= 0:
        return SizingResult(
            units=0.0,
            notional_sol=0.0,
            kelly_fraction_raw=kelly_raw,
            binding_constraint="kelly",
        )

    if notional == kelly_notional:
        binding = "kelly"
    elif notional == inputs.single_token_cap_sol:
        binding = "capital"
    else:
        binding = "hard_cap"

    units = notional / inputs.cost_per_unit_sol
    return SizingResult(
        units=units,
        notional_sol=notional,
        kelly_fraction_raw=kelly_raw,
        binding_constraint=binding,
    )
