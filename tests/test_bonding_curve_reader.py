"""Tests for the v0.3 BondingCurve account decoder.

The on-chain account layout (per pump-fun/pump-public-docs and pump-sdk
v1.32+ IDL, post the May 2025 protocol upgrade) is:

    discriminator(8) + 5 u64 LE(40) + complete bool(1) + creator pubkey(32)

The decoder also tolerates the legacy 49-byte layout (no creator field),
because some RPC responses still surface the pre-upgrade view.
"""

from __future__ import annotations

import pytest

from tsukibot_pump.solana.bonding_curve import LAMPORTS_PER_SOL
from tsukibot_pump.solana.bonding_curve_reader import (
    BONDING_CURVE_DISCRIMINATOR,
    BondingCurveDecodeError,
    decode_bonding_curve_account,
    encode_bonding_curve_account_for_tests,
)


def test_decode_full_v2_layout_returns_creator() -> None:
    """A full 81-byte v2 layout decodes virtual/real reserves + creator."""
    creator_pk = bytes(range(1, 33))  # non-zero, deterministic
    raw = encode_bonding_curve_account_for_tests(
        virtual_sol_reserves=30 * LAMPORTS_PER_SOL,
        real_sol_reserves=25 * LAMPORTS_PER_SOL,
        complete=False,
        creator=creator_pk,
    )
    acct = decode_bonding_curve_account(raw)
    assert acct.state.virtual_sol_reserves == 30 * LAMPORTS_PER_SOL
    assert acct.state.real_sol_reserves == 25 * LAMPORTS_PER_SOL
    assert acct.complete is False
    assert acct.creator is not None
    # All-zero pubkey would have decoded to None; non-zero must decode to b58.
    assert len(acct.creator) >= 32


def test_decode_complete_flag_true_when_graduated() -> None:
    raw = encode_bonding_curve_account_for_tests(complete=True)
    acct = decode_bonding_curve_account(raw)
    assert acct.complete is True


def test_decode_zero_creator_pubkey_returns_none() -> None:
    """An all-zero creator field is treated as "no creator data" (uninitialized)."""
    raw = encode_bonding_curve_account_for_tests(creator=bytes(32))
    acct = decode_bonding_curve_account(raw)
    assert acct.creator is None


def test_decode_legacy_layout_without_creator_field() -> None:
    """Pre-May-2025 accounts decode cleanly with creator=None."""
    raw = encode_bonding_curve_account_for_tests(creator=None)
    # legacy = 8 disc + 40 u64s + 1 bool = 49 bytes
    assert len(raw) == 49
    acct = decode_bonding_curve_account(raw)
    assert acct.creator is None


def test_decode_rejects_short_buffer() -> None:
    """Anything smaller than 49 bytes is rejected."""
    with pytest.raises(BondingCurveDecodeError):
        decode_bonding_curve_account(b"\x00" * 30)


def test_decode_real_sol_in_curve_uses_sol_units() -> None:
    """The convenience property converts lamports → SOL correctly."""
    raw = encode_bonding_curve_account_for_tests(real_sol_reserves=42 * LAMPORTS_PER_SOL)
    acct = decode_bonding_curve_account(raw)
    assert acct.real_sol_in_curve == pytest.approx(42.0)


def test_discriminator_constant_matches_spec() -> None:
    """The discriminator is sha256("account:BondingCurve")[:8]."""
    import hashlib

    expected = hashlib.sha256(b"account:BondingCurve").digest()[:8]
    assert expected == BONDING_CURVE_DISCRIMINATOR
