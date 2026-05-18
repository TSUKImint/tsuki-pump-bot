"""Tests for the v0.3 GraduationProbabilityScorer.

The scorer is a 4-covariate logistic regression grounded in
arXiv:2602.14860 (Marino, Naviglio, Tarantelli, Lillo, Feb 2026). The
covariates pulled from the filter stack are:

  - curve_graduation     -> bonding-curve progress (B_CURVE > 0)
  - first_kol_touch or convergence -> KOL participation (B_KOL > 0)
  - creator_vault        -> creator alignment (B_CREATOR > 0)
  - bundle_cluster       -> inverted to bot-concentration (B_BOT < 0)

We don't assert exact percentages (the coefficients are tunable); we assert
the *monotonicity* properties the paper supports:

  1. holding everything else equal, higher curve ⇒ higher P
  2. higher KOL participation ⇒ higher P
  3. higher creator alignment ⇒ higher P
  4. higher bot concentration ⇒ lower P
  5. hard-reject in any filter ⇒ score 0
"""

from __future__ import annotations

from tsukibot_pump.config import ScoringConfig, ScoringWeights
from tsukibot_pump.models import FilterOutcome
from tsukibot_pump.scoring import (
    GraduationProbabilityScorer,
    build_scorer,
)


def _config(threshold: int = 40) -> ScoringConfig:
    return ScoringConfig(
        mode="graduation_probability",
        enter_threshold=threshold,
        enter_threshold_watchtower_log=20,
        weights=ScoringWeights(
            dev_blacklist=0.20,
            bundle_cluster=0.15,
            first_kol_touch=0.20,
            convergence=0.15,
            curve_graduation=0.20,
            cto_revival=0.10,
            creator_vault=0.0,
        ),
    )


def _outcomes(
    *,
    curve: float = 50.0,
    first_kol_touch: float = 50.0,
    convergence: float = 50.0,
    creator_vault: float = 50.0,
    bundle_cluster: float = 50.0,
    dev_blacklist: float = 50.0,
    cto_revival: float = 50.0,
    hard_reject_filter: str | None = None,
) -> list[FilterOutcome]:
    out = [
        FilterOutcome(name="dev_blacklist", score=dev_blacklist),
        FilterOutcome(name="bundle_cluster", score=bundle_cluster),
        FilterOutcome(name="first_kol_touch", score=first_kol_touch),
        FilterOutcome(name="convergence", score=convergence),
        FilterOutcome(name="curve_graduation", score=curve),
        FilterOutcome(name="cto_revival", score=cto_revival),
        FilterOutcome(name="creator_vault", score=creator_vault),
    ]
    if hard_reject_filter is not None:
        for i, o in enumerate(out):
            if o.name == hard_reject_filter:
                out[i] = FilterOutcome(
                    name=o.name, score=0.0, hard_reject=True, reject_reason="test"
                )
    return out


def test_build_scorer_factory_returns_correct_type() -> None:
    scorer = build_scorer(_config())
    assert isinstance(scorer, GraduationProbabilityScorer)


def test_score_is_in_zero_to_hundred_range() -> None:
    scorer = GraduationProbabilityScorer(_config())
    composite = scorer.score(_outcomes())
    assert 0.0 <= composite.score <= 100.0


def test_hard_reject_short_circuits_to_zero() -> None:
    scorer = GraduationProbabilityScorer(_config())
    composite = scorer.score(_outcomes(hard_reject_filter="dev_blacklist"))
    assert composite.score == 0.0
    assert composite.hard_rejected is True
    assert composite.enter is False


def test_higher_curve_progress_raises_probability() -> None:
    """Monotonicity check 1: curve_graduation has positive coefficient."""
    scorer = GraduationProbabilityScorer(_config())
    low = scorer.score(_outcomes(curve=20.0))
    high = scorer.score(_outcomes(curve=95.0))
    assert high.score > low.score


def test_higher_kol_participation_raises_probability() -> None:
    """Monotonicity check 2: KOL touches have positive coefficient."""
    scorer = GraduationProbabilityScorer(_config())
    low = scorer.score(_outcomes(first_kol_touch=20.0, convergence=20.0))
    high = scorer.score(_outcomes(first_kol_touch=100.0, convergence=100.0))
    assert high.score > low.score


def test_higher_creator_alignment_raises_probability() -> None:
    """Monotonicity check 3: creator_vault has positive coefficient."""
    scorer = GraduationProbabilityScorer(_config())
    low = scorer.score(_outcomes(creator_vault=20.0))
    high = scorer.score(_outcomes(creator_vault=100.0))
    assert high.score > low.score


def test_higher_bot_concentration_lowers_probability() -> None:
    """Monotonicity check 4: bundle_cluster score↓ ⇒ bot conc↑ ⇒ P↓.

    bundle_cluster score reports the *good* axis (high = low concentration);
    the scorer inverts internally, so a low bundle score should produce a
    lower P than a high bundle score.
    """
    scorer = GraduationProbabilityScorer(_config())
    # High bundle_cluster score = LOW bot concentration → P should be higher
    high_bundle_score = scorer.score(
        _outcomes(curve=70.0, first_kol_touch=70.0, bundle_cluster=90.0)
    )
    # Low bundle_cluster score = HIGH bot concentration → P should be lower
    low_bundle_score = scorer.score(
        _outcomes(curve=70.0, first_kol_touch=70.0, bundle_cluster=10.0)
    )
    assert high_bundle_score.score > low_bundle_score.score


def test_entry_threshold_gates_correctly() -> None:
    """`composite.enter` matches the configured threshold semantics."""
    config = _config(threshold=70)
    scorer = GraduationProbabilityScorer(config)
    # Conviction inputs: high curve, high KOL, aligned creator, low bots.
    composite = scorer.score(
        _outcomes(
            curve=95.0,
            first_kol_touch=100.0,
            convergence=100.0,
            creator_vault=80.0,
            bundle_cluster=60.0,
        )
    )
    if composite.score >= 70:
        assert composite.enter is True
    else:
        assert composite.enter is False


def test_neutral_inputs_produce_low_probability() -> None:
    """All filters at 50 ⇒ score should be well below 50 (token has no edge)."""
    scorer = GraduationProbabilityScorer(_config())
    composite = scorer.score(_outcomes())
    # The model's intercept (-5) ensures a token with no signal is treated
    # as well below average. The exact value is implementation-dependent; we
    # only assert the qualitative result.
    assert composite.score < 30


def test_breakdown_records_z_and_probability() -> None:
    """Internal dashboard relies on `_grad_prob` and `_z` keys."""
    scorer = GraduationProbabilityScorer(_config())
    composite = scorer.score(_outcomes())
    assert "_grad_prob" in composite.breakdown
    assert "_z" in composite.breakdown
