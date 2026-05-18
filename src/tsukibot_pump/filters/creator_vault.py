"""Filter 7 (v0.3) — creator-vault alignment.

Background
----------
On 23 May 2025 pump.fun shipped a protocol upgrade ("Creator Rewards") that
routes 30 bps of every trade on a token to a Program-Derived Address (PDA)
keyed by the token's `creator` field. The new layout also added an explicit
`creator: Pubkey` field to every `BondingCurve` account. See
pump-fun/pump-public-docs §"Creator rewards & creator vault" and
@pump-fun/pump-sdk v1.32+ IDL.

The signal
----------
"Creator earns fees forever, even after graduation" rewires the creator
incentive from one-shot pump-and-dump to long-tail-aligned. A creator who
has shipped a graduated token *before* and is shipping a new one is using
the same pubkey to compound their vault — they're a repeat builder, not a
churn-and-burn launcher. That's a positive signal that almost no public
indexer scores on yet because most bots still look at the old "dev wallet
= signer of CREATE" instead of the new on-chain creator field.

We treat this as a *bonus* filter — never a hard reject. Anonymous /
unknown creator = neutral. Aligned creator (vault > 0 AND >=1 prior
graduations) = high score. Spammy creator (>=25 tokens in 7d with zero
graduations) = penalty. Known scammers are handled separately by
`dev_blacklist`, which still hard-rejects.

References:
  - pump.fun official fee docs (effective 7 Oct 2025): 95 bps protocol +
    30 bps creator vault = 125 bps total.
  - Marino, Naviglio, Tarantelli, Lillo (2026), arXiv:2602.14860. Section
    "creator-pubkey identity" identified as a covariate that improves
    P(graduate) prediction.
"""

from __future__ import annotations

import structlog

from ..config import CreatorVaultConfig
from ..models import FilterOutcome, TokenState

logger = structlog.get_logger(__name__)


class CreatorVaultFilter:
    """Score by creator-vault alignment + creator history."""

    NAME = "creator_vault"

    def __init__(self, config: CreatorVaultConfig) -> None:
        self.config = config

    def evaluate(self, token: TokenState) -> FilterOutcome:
        if not self.config.enabled:
            return FilterOutcome(name=self.NAME, score=50.0, notes="disabled")

        creator = token.creator
        if not creator:
            # No creator field known yet. We can't penalize tokens that
            # haven't been read from chain yet — neutral score keeps the
            # composite from over-rejecting in-flight observations.
            return FilterOutcome(
                name=self.NAME,
                score=self.config.score_anonymous,
                notes="no creator field read yet (neutral)",
            )

        prior_graduations = max(0, token.creator_prior_graduations)
        tokens_7d = max(0, token.creator_tokens_7d)
        vault_sol = max(0.0, token.creator_vault_sol)

        # Spammy creator with zero graduations → penalty, but never hard-reject.
        if tokens_7d >= self.config.suspect_if_dev_token_count_7d_gte and prior_graduations == 0:
            return FilterOutcome(
                name=self.NAME,
                score=self.config.score_suspect,
                notes=(f"creator {creator[:8]} suspect: {tokens_7d} tokens in 7d, 0 graduations"),
            )

        # Aligned creator: prior graduations on-chain ARE the signal.
        if prior_graduations >= self.config.min_prior_graduations_for_bonus:
            note = f"creator {creator[:8]} aligned: {prior_graduations} prior graduations"
            if vault_sol > 0:
                note += f", vault {vault_sol:.3f} SOL"
            return FilterOutcome(
                name=self.NAME,
                score=self.config.score_aligned,
                notes=note,
            )

        # Known creator but no prior graduations. Slight discount from
        # anonymous since we have *some* identity but no track record.
        return FilterOutcome(
            name=self.NAME,
            score=self.config.score_anonymous,
            notes=(
                f"creator {creator[:8]} new: "
                f"{prior_graduations} graduations, vault {vault_sol:.3f} SOL"
            ),
        )
