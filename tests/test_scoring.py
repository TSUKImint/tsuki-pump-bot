"""Composite scorer tests."""

from __future__ import annotations

from tsukibot_pump.config import ScoringConfig, ScoringWeights
from tsukibot_pump.models import FilterOutcome
from tsukibot_pump.scoring import CompositeScorer


def _config() -> ScoringConfig:
    return ScoringConfig(
        enter_threshold=60,
        enter_threshold_watchtower_log=40,
        weights=ScoringWeights(
            dev_blacklist=0.20,
            bundle_cluster=0.15,
            first_kol_touch=0.20,
            convergence=0.15,
            curve_graduation=0.20,
            cto_revival=0.10,
        ),
    )


def _outcomes(score: float) -> list[FilterOutcome]:
    """Return one outcome per filter at the given score."""
    return [
        FilterOutcome(name="dev_blacklist", score=score),
        FilterOutcome(name="bundle_cluster", score=score),
        FilterOutcome(name="first_kol_touch", score=score),
        FilterOutcome(name="convergence", score=score),
        FilterOutcome(name="curve_graduation", score=score),
        FilterOutcome(name="cto_revival", score=score),
    ]


def test_uniform_score_yields_same_composite() -> None:
    scorer = CompositeScorer(_config())
    composite = scorer.score(_outcomes(80))
    assert composite.score == 80.0
    assert composite.enter is True


def test_below_threshold_does_not_enter() -> None:
    scorer = CompositeScorer(_config())
    composite = scorer.score(_outcomes(40))
    assert composite.enter is False


def test_hard_reject_short_circuits_to_zero() -> None:
    scorer = CompositeScorer(_config())
    outcomes = _outcomes(100)
    outcomes[0] = FilterOutcome(
        name="dev_blacklist",
        score=0.0,
        hard_reject=True,
        reject_reason="known rugger",
    )
    composite = scorer.score(outcomes)
    assert composite.hard_rejected
    assert composite.enter is False
    assert composite.score == 0.0
    assert "dev_blacklist" in composite.reject_reason


def test_weighted_average_matches_manual() -> None:
    scorer = CompositeScorer(_config())
    outcomes = [
        FilterOutcome(name="dev_blacklist", score=100),
        FilterOutcome(name="bundle_cluster", score=100),
        FilterOutcome(name="first_kol_touch", score=0),
        FilterOutcome(name="convergence", score=0),
        FilterOutcome(name="curve_graduation", score=100),
        FilterOutcome(name="cto_revival", score=0),
    ]
    composite = scorer.score(outcomes)
    # = 0.20*100 + 0.15*100 + 0.20*0 + 0.15*0 + 0.20*100 + 0.10*0
    # = 20 + 15 + 0 + 0 + 20 + 0 = 55
    assert abs(composite.score - 55.0) < 1e-6


def test_breakdown_contains_each_filter() -> None:
    scorer = CompositeScorer(_config())
    composite = scorer.score(_outcomes(70))
    assert set(composite.breakdown.keys()) == {
        "dev_blacklist",
        "bundle_cluster",
        "first_kol_touch",
        "convergence",
        "curve_graduation",
        "cto_revival",
    }
