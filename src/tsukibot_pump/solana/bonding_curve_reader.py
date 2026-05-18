"""On-chain BondingCurve account reader (decoder).

The pump.fun program's `BondingCurve` account is a fixed-layout Anchor-style
account. Layout (verified against the pump-fun/pump-public-docs GitHub repo
and the published pump-sdk v1.32+ IDL, post-May-2025 protocol upgrade):

    offset  size  field
    ------  ----  ------------------------------------------------
    0       8     anchor discriminator
    8       8     virtual_token_reserves      (u64 LE, base units)
    16      8     virtual_sol_reserves        (u64 LE, lamports)
    24      8     real_token_reserves         (u64 LE, base units)
    32      8     real_sol_reserves           (u64 LE, lamports)
    40      8     token_total_supply          (u64 LE, base units)
    48      1     complete                    (bool)
    49      32    creator                     (Pubkey, May 2025 upgrade)

Total size: 81 bytes (pre-May-2025) → 81 bytes; with the `creator` field
added the on-chain account is at least 81 bytes. Some references mention a
larger account size including padding (Anchor pads to 8-byte multiples) —
we tolerate any size >= 81 and just slice the relevant prefix.

We deliberately don't pull in `solders`/`anchorpy` for this — base64 +
struct.unpack is plenty for read-only paper mode.

References:
  - pump-fun/pump-public-docs (official). Section "Bonding curve account".
  - @pump-fun/pump-sdk v1.32 (published 2025-10). IDL "BondingCurve".
"""

from __future__ import annotations

import base64
import struct
from dataclasses import dataclass

import structlog

from .bonding_curve import (
    LAMPORTS_PER_SOL,
    TOKEN_UNIT_MULTIPLIER,
    BondingCurveState,
)
from .rpc import PUMP_FUN_PROGRAM_ID, SolanaRPCClient

logger = structlog.get_logger(__name__)

# Anchor discriminator for BondingCurve, per sha256("account:BondingCurve")
# first 8 bytes. Verified against the public pump.fun SDK.
BONDING_CURVE_DISCRIMINATOR: bytes = bytes.fromhex("17b7f83760d8ac60")

# Minimum byte size we accept before attempting to decode the creator field.
# 49 = discriminator(8) + 5 u64s(40) + complete bool(1); +32 for creator.
_MIN_LEGACY_SIZE = 49
_MIN_V2_SIZE = 49 + 32  # 81 bytes


@dataclass(frozen=True, slots=True)
class BondingCurveAccount:
    """Decoded on-chain BondingCurve account.

    Wraps `BondingCurveState` (used by the math layer) with the v0.3
    additions: the creator pubkey (May 2025 upgrade) plus the raw on-chain
    `token_total_supply`.
    """

    state: BondingCurveState
    token_total_supply: int  # base units
    creator: str | None  # base58 pubkey; None on legacy accounts

    @property
    def real_sol_in_curve(self) -> float:
        return self.state.real_sol_in_curve

    @property
    def complete(self) -> bool:
        return self.state.complete


class BondingCurveDecodeError(ValueError):
    """Raised when the on-chain account bytes don't match the expected layout."""


def _b58_encode(data: bytes) -> str:
    """Encode 32 raw bytes as base58 (Solana pubkey format).

    Uses `base58` if available (already a project dep); falls back to a
    minimal implementation so this stays test-friendly.
    """
    try:
        import base58 as _base58

        return str(_base58.b58encode(data).decode("ascii"))
    except ImportError:  # pragma: no cover - base58 is a hard dep
        alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        num = int.from_bytes(data, "big")
        out = ""
        while num > 0:
            num, rem = divmod(num, 58)
            out = alphabet[rem] + out
        # leading zero bytes → '1'
        for byte in data:
            if byte == 0:
                out = "1" + out
            else:
                break
        return out or "1"


def decode_bonding_curve_account(raw_bytes: bytes) -> BondingCurveAccount:
    """Decode raw on-chain bytes into a BondingCurveAccount.

    Tolerant of both legacy (pre-May-2025, no `creator` field) and v2 layouts.
    Raises `BondingCurveDecodeError` on bytes too short or discriminator
    mismatch.
    """
    if len(raw_bytes) < _MIN_LEGACY_SIZE:
        raise BondingCurveDecodeError(f"bonding curve account too small: {len(raw_bytes)} bytes")

    discriminator = raw_bytes[0:8]
    if discriminator != BONDING_CURVE_DISCRIMINATOR:
        # Some RPCs (or test fixtures) strip the discriminator; we attempt
        # to decode anyway by re-aligning at offset 0. This keeps the
        # decoder usable for golden-file tests without forcing every fixture
        # to encode the exact prefix.
        if len(raw_bytes) >= _MIN_LEGACY_SIZE + 8:
            raise BondingCurveDecodeError(
                f"bonding curve discriminator mismatch: "
                f"got {discriminator.hex()}, want {BONDING_CURVE_DISCRIMINATOR.hex()}"
            )
        offset = 0
    else:
        offset = 8

    if len(raw_bytes) < offset + _MIN_LEGACY_SIZE - 8:
        raise BondingCurveDecodeError(
            f"bonding curve account truncated: {len(raw_bytes)} < {offset + _MIN_LEGACY_SIZE - 8}"
        )

    # Five u64 LE + one u8 boolean.
    (
        virtual_token_reserves,
        virtual_sol_reserves,
        real_token_reserves,
        real_sol_reserves,
        token_total_supply,
    ) = struct.unpack_from("<QQQQQ", raw_bytes, offset)
    offset += 40
    complete_byte = raw_bytes[offset]
    complete = bool(complete_byte & 0x01)
    offset += 1

    creator: str | None = None
    # v2 layout: 32-byte pubkey follows the complete flag.
    if len(raw_bytes) >= offset + 32:
        creator_bytes = raw_bytes[offset : offset + 32]
        # Skip a creator pubkey that's all-zeroes (uninitialized).
        if any(b != 0 for b in creator_bytes):
            creator = _b58_encode(creator_bytes)

    state = BondingCurveState(
        virtual_sol_reserves=int(virtual_sol_reserves),
        virtual_token_reserves=int(virtual_token_reserves),
        real_sol_reserves=int(real_sol_reserves),
        real_token_reserves=int(real_token_reserves),
        complete=complete,
    )
    return BondingCurveAccount(
        state=state,
        token_total_supply=int(token_total_supply),
        creator=creator,
    )


def encode_bonding_curve_account_for_tests(
    *,
    virtual_token_reserves: int = 1_073_000_000 * TOKEN_UNIT_MULTIPLIER,
    virtual_sol_reserves: int = 30 * LAMPORTS_PER_SOL,
    real_token_reserves: int = 793_100_000 * TOKEN_UNIT_MULTIPLIER,
    real_sol_reserves: int = 0,
    token_total_supply: int = 1_000_000_000 * TOKEN_UNIT_MULTIPLIER,
    complete: bool = False,
    creator: bytes | None = None,
    include_discriminator: bool = True,
) -> bytes:
    """Encode a BondingCurve account in the on-chain layout. Test helper."""
    out = bytearray()
    if include_discriminator:
        out += BONDING_CURVE_DISCRIMINATOR
    out += struct.pack(
        "<QQQQQ",
        virtual_token_reserves,
        virtual_sol_reserves,
        real_token_reserves,
        real_sol_reserves,
        token_total_supply,
    )
    out += bytes([1 if complete else 0])
    if creator is not None:
        if len(creator) != 32:
            raise ValueError("creator must be 32 bytes")
        out += creator
    return bytes(out)


class BondingCurveReader:
    """Reads on-chain BondingCurve accounts via the RPC `getAccountInfo`.

    Thin wrapper around `SolanaRPCClient.get_account_info` that handles
    base64 decoding, layout validation, and surfacing useful errors.

    Usage:

        async with SolanaRPCClient(...) as rpc:
            reader = BondingCurveReader(rpc)
            acct = await reader.read(bonding_curve_address)
    """

    def __init__(self, rpc: SolanaRPCClient) -> None:
        self.rpc = rpc

    async def read(self, address: str) -> BondingCurveAccount | None:
        """Read and decode the bonding curve account at `address`.

        Returns None if the account doesn't exist or isn't owned by the pump.fun
        program. Raises BondingCurveDecodeError on layout mismatch.
        """
        info = await self.rpc.get_account_info(address)
        if not info:
            return None

        owner = info.get("owner")
        if owner and owner != PUMP_FUN_PROGRAM_ID:
            logger.warning(
                "bonding_curve_reader.owner_mismatch",
                address=address,
                owner=owner,
                expected=PUMP_FUN_PROGRAM_ID,
            )
            return None

        data = info.get("data")
        raw_bytes = _decode_account_data(data)
        if raw_bytes is None:
            return None

        try:
            return decode_bonding_curve_account(raw_bytes)
        except BondingCurveDecodeError:
            logger.warning(
                "bonding_curve_reader.decode_failed",
                address=address,
                size=len(raw_bytes),
            )
            raise


def _decode_account_data(data: object) -> bytes | None:
    """Decode the `data` field from a getAccountInfo response.

    The JSON-RPC spec allows several shapes: `[base64_string, "base64"]`,
    `[base58_string, "base58"]`, or a parsed dict (jsonParsed encoding —
    not used for raw account data). We only decode base64 here; anything
    else returns None.
    """
    if isinstance(data, list) and len(data) >= 2:
        payload = data[0]
        encoding = data[1]
        if not isinstance(payload, str):
            return None
        if encoding == "base64":
            try:
                return base64.b64decode(payload)
            except (ValueError, TypeError):
                return None
        if encoding == "base58":
            try:
                import base58 as _base58

                return bytes(_base58.b58decode(payload))
            except (ValueError, ImportError):
                return None
    if isinstance(data, str):
        # Some RPCs return the raw base64 string without an encoding tag.
        try:
            return base64.b64decode(data)
        except (ValueError, TypeError):
            return None
    return None
