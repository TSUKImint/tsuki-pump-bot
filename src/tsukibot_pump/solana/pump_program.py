"""Pump.fun program data types + light transaction parsing.

We only need to:
  - identify pump.fun create / buy / sell instructions in a jsonParsed tx
  - extract: mint, dev wallet (signer of create), buyer wallet, SOL amount,
    token amount, slot, blockTime

Heavyweight Anchor IDL parsing is intentionally avoided — we lean on the
`jsonParsed` encoding returned by mainstream RPCs, which already decodes
SPL token transfers and surfaces program instruction names by program ID.

The instruction discriminators below come from the published IDL
(https://github.com/rckprtr/pumpdotfun-sdk). They're 8-byte SHA256 prefixes
encoded into the instruction data. We only need them when we get a binary-
encoded instruction; jsonParsed callers can match by program ID + index.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .rpc import PUMP_FUN_PROGRAM_ID


class PumpInstructionKind(StrEnum):
    CREATE = "create"
    BUY = "buy"
    SELL = "sell"
    WITHDRAW = "withdraw"
    UNKNOWN = "unknown"


# 8-byte Anchor discriminators (first 8 bytes of sha256("global:<method>"))
# Hex-encoded. Verified against the public pumpdotfun-sdk IDL.
_DISCRIMINATORS: dict[bytes, PumpInstructionKind] = {
    bytes.fromhex("181ec828051c0777"): PumpInstructionKind.CREATE,
    bytes.fromhex("66063d1201daebea"): PumpInstructionKind.BUY,
    bytes.fromhex("33e685a4017f83ad"): PumpInstructionKind.SELL,
    bytes.fromhex("b712469c946da122"): PumpInstructionKind.WITHDRAW,
}


def classify_instruction_data(data: bytes) -> PumpInstructionKind:
    """Classify a pump.fun instruction by its first 8 bytes."""
    if len(data) < 8:
        return PumpInstructionKind.UNKNOWN
    return _DISCRIMINATORS.get(data[:8], PumpInstructionKind.UNKNOWN)


@dataclass(frozen=True, slots=True)
class PumpEvent:
    """High-level event extracted from a transaction."""

    kind: PumpInstructionKind
    signature: str
    slot: int
    block_time_unix: int | None
    mint: str | None
    dev_wallet: str | None
    actor_wallet: str | None  # buyer / seller / withdrawer
    sol_amount: float | None  # SOL involved (positive = into curve)
    token_amount: float | None  # token units (positive = into actor)


def _extract_account(account_keys: list[Any], idx: int) -> str | None:
    """Pull a base58 pubkey from a (possibly typed) account_keys list."""
    if idx >= len(account_keys):
        return None
    entry = account_keys[idx]
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        pubkey = entry.get("pubkey")
        return str(pubkey) if pubkey else None
    return None


def parse_pump_instructions(tx: dict[str, Any]) -> list[PumpEvent]:
    """Extract pump.fun events from a `jsonParsed` getTransaction response.

    Tolerant of partial / malformed responses — returns empty list rather
    than raising. The caller filters by `kind`.
    """
    if not tx:
        return []
    meta = tx.get("meta") or {}
    transaction = tx.get("transaction") or {}
    message = transaction.get("message") or {}
    instructions = list(message.get("instructions") or [])
    inner_instructions = list(meta.get("innerInstructions") or [])
    account_keys = message.get("accountKeys") or []
    slot = int(tx.get("slot", 0) or 0)
    block_time = tx.get("blockTime")
    block_time_unix = int(block_time) if block_time is not None else None
    signatures = transaction.get("signatures") or []
    signature = signatures[0] if signatures else ""

    events: list[PumpEvent] = []

    def _record(ix: dict[str, Any]) -> None:
        program_id = ix.get("programId") or _extract_account(
            account_keys, int(ix.get("programIdIndex", -1))
        )
        if program_id != PUMP_FUN_PROGRAM_ID:
            return

        kind: PumpInstructionKind
        parsed = ix.get("parsed")
        if isinstance(parsed, dict) and parsed.get("type"):
            ptype = str(parsed.get("type", "")).lower()
            try:
                kind = PumpInstructionKind(ptype)
            except ValueError:
                kind = PumpInstructionKind.UNKNOWN
        else:
            data_str = ix.get("data") or ""
            try:
                import base58

                raw = base58.b58decode(data_str)
            except (ValueError, ImportError):
                raw = b""
            kind = classify_instruction_data(raw) if raw else PumpInstructionKind.UNKNOWN

        # Heuristic actor extraction from accounts list. The pump.fun IDL
        # ordering is documented but RPCs sometimes flatten it; we walk what
        # we have.
        accounts = ix.get("accounts") or []
        mint: str | None = None
        actor: str | None = None
        dev_wallet: str | None = None

        if accounts:
            # In CREATE: first account is the mint, the signer (dev) is in
            # message-level account_keys[0] typically.
            if kind == PumpInstructionKind.CREATE:
                mint = _extract_account(accounts, 0) or _extract_account(account_keys, 0)
                dev_wallet = _extract_account(account_keys, 0)
                actor = dev_wallet
            else:
                # BUY / SELL: per IDL, accounts[2] = mint, accounts[7] = user.
                mint = _extract_account(accounts, 2) or _extract_account(accounts, 0)
                actor = _extract_account(accounts, 7) or _extract_account(account_keys, 0)

        events.append(
            PumpEvent(
                kind=kind,
                signature=str(signature),
                slot=slot,
                block_time_unix=block_time_unix,
                mint=mint,
                dev_wallet=dev_wallet,
                actor_wallet=actor,
                sol_amount=None,  # filled in by caller via meta.pre/postBalances diff
                token_amount=None,  # filled in by caller via meta.pre/postTokenBalances
            )
        )

    for ix in instructions:
        if isinstance(ix, dict):
            _record(ix)
    for inner in inner_instructions:
        for ix in inner.get("instructions") or []:
            if isinstance(ix, dict):
                _record(ix)

    return events


def sol_delta_for_wallet(tx: dict[str, Any], wallet: str) -> float | None:
    """Compute net SOL change for a given wallet from a transaction.

    Returns SOL (signed: positive = wallet received SOL). None if the wallet
    isn't found in the transaction's account list.
    """
    meta = tx.get("meta") or {}
    transaction = tx.get("transaction") or {}
    message = transaction.get("message") or {}
    account_keys = message.get("accountKeys") or []
    pre = meta.get("preBalances") or []
    post = meta.get("postBalances") or []

    for idx, entry in enumerate(account_keys):
        key = (
            entry
            if isinstance(entry, str)
            else (entry.get("pubkey") if isinstance(entry, dict) else None)
        )
        if key == wallet and idx < len(pre) and idx < len(post):
            return (int(post[idx]) - int(pre[idx])) / 1_000_000_000
    return None
