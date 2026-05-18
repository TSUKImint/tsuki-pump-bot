"""Tests for the v0.3 CreatorVaultFilter (Filter 7).

The filter never hard-rejects. It only nudges the composite score based on
on-chain creator history. Three outcomes:

  - aligned (>= min_prior_graduations_for_bonus prior wins) → score_aligned (80)
  - anonymous (no creator field, OR known creator w/o prior wins) → score_anonymous (50)
  - suspect (>= suspect_if_dev_token_count_7d_gte tokens, 0 graduations) → score_suspect (20)
"""

from __future__ import annotations

from tsukibot_pump.config import CreatorVaultConfig
from tsukibot_pump.filters import CreatorVaultFilter
from tsukibot_pump.models import TokenState


def _config(**overrides: object) -> CreatorVaultConfig:
    base = {
        "enabled": True,
        "min_prior_graduations_for_bonus": 1,
        "suspect_if_dev_token_count_7d_gte": 25,
        "score_aligned": 80.0,
        "score_anonymous": 50.0,
        "score_suspect": 20.0,
    }
    base.update(overrides)
    return CreatorVaultConfig(**base)  # type: ignore[arg-type]


def _token(**overrides: object) -> TokenState:
    base = {
        "mint": "MintXXX",
        "creator": "CreatorPubkeyExample" + "X" * 12,
        "creator_vault_sol": 0.0,
        "creator_prior_graduations": 0,
        "creator_tokens_7d": 0,
    }
    base.update(overrides)
    return TokenState(**base)  # type: ignore[arg-type]


def test_disabled_returns_neutral() -> None:
    f = CreatorVaultFilter(_config(enabled=False))
    out = f.evaluate(_token())
    assert out.score == 50.0
    assert "disabled" in out.notes


def test_no_creator_returns_anonymous() -> None:
    """Unknown creator pubkey ⇒ neutral; never penalize what we haven't read."""
    f = CreatorVaultFilter(_config())
    out = f.evaluate(_token(creator=None))
    assert out.score == 50.0
    assert "neutral" in out.notes or "no creator" in out.notes.lower()


def test_aligned_creator_with_prior_graduation_gets_bonus() -> None:
    """The flagship v0.3 signal: prior on-chain graduation."""
    f = CreatorVaultFilter(_config())
    out = f.evaluate(_token(creator_prior_graduations=3, creator_vault_sol=4.2))
    assert out.score == 80.0
    assert "aligned" in out.notes
    assert "3 prior graduations" in out.notes


def test_known_creator_with_zero_graduations_returns_anonymous() -> None:
    """Known creator pubkey but no track record ⇒ anonymous tier, not aligned."""
    f = CreatorVaultFilter(_config())
    out = f.evaluate(_token(creator_prior_graduations=0, creator_tokens_7d=2))
    assert out.score == 50.0
    assert "0 graduations" in out.notes


def test_suspect_creator_many_tokens_zero_graduations_gets_penalty() -> None:
    """Spam-launcher with zero graduations ⇒ score_suspect (20)."""
    f = CreatorVaultFilter(_config())
    out = f.evaluate(_token(creator_prior_graduations=0, creator_tokens_7d=30))
    assert out.score == 20.0
    assert "suspect" in out.notes


def test_suspect_creator_with_prior_graduations_falls_back_to_aligned() -> None:
    """A high token count is forgiven if creator has graduated something."""
    f = CreatorVaultFilter(_config())
    out = f.evaluate(_token(creator_prior_graduations=2, creator_tokens_7d=30))
    assert out.score == 80.0
    assert "aligned" in out.notes


def test_filter_never_hard_rejects() -> None:
    """Even worst-case (suspect creator), the filter doesn't hard-reject."""
    f = CreatorVaultFilter(_config())
    out = f.evaluate(_token(creator_prior_graduations=0, creator_tokens_7d=100))
    assert out.score == 20.0
    assert out.hard_reject is False
