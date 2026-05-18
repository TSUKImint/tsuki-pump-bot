"""pump.fun program event parser tests."""

from __future__ import annotations

from tsukibot_pump.solana.pump_program import (
    PumpInstructionKind,
    classify_instruction_data,
    parse_pump_instructions,
    sol_delta_for_wallet,
)
from tsukibot_pump.solana.rpc import PUMP_FUN_PROGRAM_ID


def test_classify_buy_discriminator() -> None:
    raw = bytes.fromhex("66063d1201daebea") + b"\x00" * 16
    assert classify_instruction_data(raw) == PumpInstructionKind.BUY


def test_classify_sell_discriminator() -> None:
    raw = bytes.fromhex("33e685a4017f83ad")
    assert classify_instruction_data(raw) == PumpInstructionKind.SELL


def test_classify_unknown_returns_unknown() -> None:
    raw = b"\x00" * 8
    assert classify_instruction_data(raw) == PumpInstructionKind.UNKNOWN


def test_classify_short_data_returns_unknown() -> None:
    assert classify_instruction_data(b"abc") == PumpInstructionKind.UNKNOWN


def test_parse_returns_empty_on_no_instructions() -> None:
    assert parse_pump_instructions({}) == []
    assert parse_pump_instructions({"meta": {}, "transaction": {}}) == []


def test_parse_buy_via_parsed_type() -> None:
    """When the RPC returns a parsed instruction (rare for custom programs),
    we should still extract it correctly."""
    tx = {
        "slot": 12345,
        "blockTime": 1_700_000_000,
        "transaction": {
            "signatures": ["abc123"],
            "message": {
                "accountKeys": [
                    "BuyerWallet000000000000000000000000000000000",
                    "AccB",
                    "MintAccount111111111111111111111111111111111",
                    "AccD",
                    "AccE",
                    "AccF",
                    "AccG",
                    "ActorWallet22222222222222222222222222222222",
                ],
                "instructions": [
                    {
                        "programId": PUMP_FUN_PROGRAM_ID,
                        "parsed": {"type": "buy", "info": {}},
                        "accounts": [
                            "AccA",
                            "AccB",
                            "MintAccount111111111111111111111111111111111",
                            "AccD",
                            "AccE",
                            "AccF",
                            "AccG",
                            "ActorWallet22222222222222222222222222222222",
                        ],
                    }
                ],
            },
        },
        "meta": {},
    }
    events = parse_pump_instructions(tx)
    assert len(events) == 1
    assert events[0].kind == PumpInstructionKind.BUY
    assert events[0].signature == "abc123"
    assert events[0].slot == 12345
    assert events[0].block_time_unix == 1_700_000_000
    assert events[0].mint == "MintAccount111111111111111111111111111111111"


def test_parse_skips_non_pump_program_instructions() -> None:
    tx = {
        "transaction": {
            "signatures": ["sig"],
            "message": {
                "accountKeys": ["A", "B"],
                "instructions": [
                    {"programId": "SomeOtherProgram111111", "parsed": {"type": "buy"}}
                ],
            },
        },
        "meta": {},
    }
    assert parse_pump_instructions(tx) == []


def test_parse_handles_inner_instructions() -> None:
    tx = {
        "slot": 1,
        "transaction": {
            "signatures": ["s"],
            "message": {
                "accountKeys": ["A"],
                "instructions": [],
            },
        },
        "meta": {
            "innerInstructions": [
                {
                    "index": 0,
                    "instructions": [
                        {
                            "programId": PUMP_FUN_PROGRAM_ID,
                            "parsed": {"type": "sell"},
                            "accounts": [
                                "Z",
                                "Y",
                                "MintInner111111111111111111111111111",
                                "X",
                                "W",
                                "V",
                                "U",
                                "ActorInner1111111111111111111111111",
                            ],
                        }
                    ],
                }
            ]
        },
    }
    events = parse_pump_instructions(tx)
    assert len(events) == 1
    assert events[0].kind == PumpInstructionKind.SELL


def test_sol_delta_for_wallet_returns_signed_delta() -> None:
    tx = {
        "transaction": {
            "message": {
                "accountKeys": ["WalletA", "WalletB"],
            }
        },
        "meta": {
            "preBalances": [1_000_000_000, 500_000_000],
            "postBalances": [500_000_000, 1_000_000_000],
        },
    }
    assert sol_delta_for_wallet(tx, "WalletA") == -0.5
    assert sol_delta_for_wallet(tx, "WalletB") == 0.5
    assert sol_delta_for_wallet(tx, "UnknownWallet") is None
