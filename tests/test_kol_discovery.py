"""End-to-end + unit tests for the KOL discovery package.

Covers:
* outcome classification (graduated / winner / dead / pending)
* wallet feature extraction (hit rate, log-ROI, recency, CVs)
* bot heuristics (sniper, mechanical size, high-frequency, mechanical hold)
* poison heuristics (own funder = dev, shared funder, dump-bus)
* full pipeline: synthetic firehose → top-N CSV
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsukibot_pump.discovery import (
    BotHeuristicsConfig,
    KolDiscoveryConfig,
    PoisonHeuristicsConfig,
    TokenOutcomeLabel,
    build_wallet_features,
    classify_token_outcomes,
    discover_kols,
    is_likely_bot,
    is_likely_poison_wallet,
)
from tsukibot_pump.discovery.kol_discovery import write_kol_csv
from tsukibot_pump.discovery.outcomes import TokenOutcome
from tsukibot_pump.solana.pump_program import PumpEvent, PumpInstructionKind

# ── helpers ───────────────────────────────────────────────────────────────


def _create(mint: str, dev: str, slot: int = 100, ts: int = 1_000_000) -> PumpEvent:
    return PumpEvent(
        kind=PumpInstructionKind.CREATE,
        signature=f"sig-create-{mint}",
        slot=slot,
        block_time_unix=ts,
        mint=mint,
        dev_wallet=dev,
        actor_wallet=dev,
        sol_amount=None,
        token_amount=None,
    )


def _buy(
    mint: str,
    wallet: str,
    *,
    slot: int,
    ts: int,
    sol: float,
    tokens: float,
    dev: str | None = None,
) -> PumpEvent:
    return PumpEvent(
        kind=PumpInstructionKind.BUY,
        signature=f"sig-buy-{mint}-{wallet}-{slot}",
        slot=slot,
        block_time_unix=ts,
        mint=mint,
        dev_wallet=dev,
        actor_wallet=wallet,
        sol_amount=sol,
        token_amount=tokens,
    )


def _sell(
    mint: str,
    wallet: str,
    *,
    slot: int,
    ts: int,
    sol: float,
    tokens: float,
) -> PumpEvent:
    return PumpEvent(
        kind=PumpInstructionKind.SELL,
        signature=f"sig-sell-{mint}-{wallet}-{slot}",
        slot=slot,
        block_time_unix=ts,
        mint=mint,
        dev_wallet=None,
        actor_wallet=wallet,
        sol_amount=sol,
        token_amount=tokens,
    )


# ── outcomes ──────────────────────────────────────────────────────────────


def test_classify_winner_token() -> None:
    """Token that goes from price 0.001 → 0.005 (5x) classifies as WINNER."""
    events = [
        _create("MINT_A", dev="DEV_A"),
        _buy("MINT_A", "WHALE_1", slot=110, ts=1_000_100, sol=1.0, tokens=1000.0),  # price=0.001
        _buy("MINT_A", "WHALE_2", slot=120, ts=1_000_200, sol=5.0, tokens=1000.0),  # price=0.005
    ]
    outcomes = classify_token_outcomes(events, min_multiple_winner=3.0)
    assert outcomes["MINT_A"].label is TokenOutcomeLabel.WINNER
    assert outcomes["MINT_A"].peak_multiple == pytest.approx(5.0)


def test_classify_dead_token() -> None:
    """Token that barely moves classifies as DEAD."""
    events = [
        _create("MINT_B", dev="DEV_B"),
        _buy("MINT_B", "W1", slot=110, ts=1_000_100, sol=1.0, tokens=1000.0),
        _buy("MINT_B", "W2", slot=120, ts=1_000_200, sol=1.05, tokens=1000.0),  # 1.05x
    ]
    outcomes = classify_token_outcomes(events, min_multiple_winner=3.0)
    assert outcomes["MINT_B"].label is TokenOutcomeLabel.DEAD


def test_classify_graduated_token() -> None:
    """A token that accumulates >= 85 SOL classifies as GRADUATED."""
    events = [_create("MINT_C", dev="DEV_C")]
    # 95 buys of 1 SOL each → 95 SOL accumulated, well over graduation.
    for i in range(95):
        events.append(
            _buy(
                "MINT_C",
                f"W{i}",
                slot=110 + i,
                ts=1_000_100 + i * 60,
                sol=1.0,
                tokens=1000.0 * (1 + 0.01 * i),  # rising price as curve fills
            )
        )
    outcomes = classify_token_outcomes(events)
    assert outcomes["MINT_C"].label is TokenOutcomeLabel.GRADUATED
    assert outcomes["MINT_C"].last_real_sol_in_curve >= 85.0


def test_classify_pending_token_with_now_unix() -> None:
    """A token within its observation window is PENDING and untreated."""
    events = [
        _create("MINT_D", dev="DEV_D", ts=2_000_000),
        _buy("MINT_D", "W1", slot=110, ts=2_000_100, sol=1.0, tokens=1000.0),
        _buy("MINT_D", "W2", slot=120, ts=2_000_200, sol=10.0, tokens=1000.0),
    ]
    outcomes = classify_token_outcomes(events, now_unix=2_000_500)
    # Less than 24h elapsed since first_seen → PENDING.
    assert outcomes["MINT_D"].label is TokenOutcomeLabel.PENDING


# ── wallet features ───────────────────────────────────────────────────────


def test_wallet_features_hit_rate_excludes_pending() -> None:
    """Hit rate denominator must exclude pending tokens — otherwise we
    punish wallets who recently bought tokens still inside their window.
    """
    winner = TokenOutcome(
        mint="MINT_W",
        dev_wallet="DEV_W",
        first_seen_unix=1_000_000,
        first_price_sol_per_token=0.001,
        peak_price_sol_per_token=0.005,
        peak_multiple=5.0,
        last_real_sol_in_curve=10.0,
        label=TokenOutcomeLabel.WINNER,
        early_buyer_wallets=("W",),
    )
    dead = TokenOutcome(
        mint="MINT_D",
        dev_wallet="DEV_D",
        first_seen_unix=1_000_000,
        first_price_sol_per_token=0.001,
        peak_price_sol_per_token=0.001,
        peak_multiple=1.0,
        last_real_sol_in_curve=0.5,
        label=TokenOutcomeLabel.DEAD,
        early_buyer_wallets=("W",),
    )
    pending = TokenOutcome(
        mint="MINT_P",
        dev_wallet="DEV_P",
        first_seen_unix=2_000_000,
        first_price_sol_per_token=0.001,
        peak_price_sol_per_token=0.002,
        peak_multiple=2.0,
        last_real_sol_in_curve=2.0,
        label=TokenOutcomeLabel.PENDING,
        early_buyer_wallets=("W",),
    )
    outcomes = {o.mint: o for o in (winner, dead, pending)}

    events = [
        _buy("MINT_W", "W", slot=110, ts=1_000_100, sol=1.0, tokens=1000.0),
        _buy("MINT_D", "W", slot=120, ts=1_000_200, sol=1.0, tokens=1000.0),
        _buy("MINT_P", "W", slot=130, ts=2_000_100, sol=1.0, tokens=1000.0),
    ]
    feats = build_wallet_features(events, outcomes)
    feat = feats["W"]
    # 1 winner / (1 winner + 1 dead) = 50%, pending excluded from denominator
    assert feat.hit_rate == pytest.approx(0.5)
    assert feat.n_trades_on_pending == 1


def test_wallet_features_log_roi_geometric_mean() -> None:
    """log-ROI must be the *geometric* mean — a single 100x can't dominate."""
    outcomes = {
        f"MINT_{i}": TokenOutcome(
            mint=f"MINT_{i}",
            dev_wallet=f"DEV_{i}",
            first_seen_unix=1_000_000,
            first_price_sol_per_token=0.001,
            peak_price_sol_per_token=peak,
            peak_multiple=peak / 0.001,
            last_real_sol_in_curve=2.0,
            label=TokenOutcomeLabel.WINNER,
            early_buyer_wallets=("WALLET",),
        )
        for i, peak in enumerate([0.003, 0.004, 0.005, 0.100])  # 3x, 4x, 5x, 100x
    }
    events = [
        _buy(f"MINT_{i}", "WALLET", slot=110 + i, ts=1_000_100 + i * 60, sol=1.0, tokens=1000.0)
        for i in range(4)
    ]
    feats = build_wallet_features(events, outcomes)
    feat = feats["WALLET"]
    # geometric mean of (3, 4, 5, 100) ≈ 7.4 — much less than the
    # arithmetic mean (28). That's the whole point.
    import math

    expected = (math.log(3) + math.log(4) + math.log(5) + math.log(100)) / 4
    assert feat.log_roi_mean == pytest.approx(expected, rel=1e-3)


def test_wallet_features_tracks_hold_times_for_roundtrips() -> None:
    events = [
        _create("MINT_X", dev="DEV_X"),
        _buy("MINT_X", "W", slot=110, ts=1_000_100, sol=1.0, tokens=1000.0),
        _sell("MINT_X", "W", slot=200, ts=1_000_300, sol=2.0, tokens=1000.0),
        _create("MINT_Y", dev="DEV_Y", slot=300, ts=1_001_000),
        _buy("MINT_Y", "W", slot=310, ts=1_001_100, sol=1.0, tokens=1000.0),
        _sell("MINT_Y", "W", slot=400, ts=1_001_500, sol=2.0, tokens=1000.0),
    ]
    feats = build_wallet_features(events, outcomes={})
    feat = feats["W"]
    assert feat.n_observed_round_trips == 2
    assert feat.hold_times_seconds == [200.0, 400.0]


# ── bot heuristics ────────────────────────────────────────────────────────


def test_bot_flag_same_slot_sniper() -> None:
    from tsukibot_pump.discovery.wallet_features import WalletFeatures

    feat = WalletFeatures(wallet="BOT")
    feat.n_observed_buys = 10
    feat.same_slot_entries = 6  # 60% same-slot
    feat.first_seen_unix = 1_000_000
    feat.last_seen_unix = 1_086_400  # 1 day
    flagged, reason = is_likely_bot(feat, BotHeuristicsConfig())
    assert flagged
    assert "same-slot" in reason


def test_bot_flag_high_frequency() -> None:
    from tsukibot_pump.discovery.wallet_features import WalletFeatures

    feat = WalletFeatures(wallet="BOT")
    feat.n_observed_buys = 500
    feat.first_seen_unix = 1_000_000
    feat.last_seen_unix = 1_086_400  # 1 day → 500/day
    flagged, reason = is_likely_bot(feat, BotHeuristicsConfig(max_trades_per_day=80.0))
    assert flagged
    assert "frequency" in reason


def test_bot_flag_mechanical_size() -> None:
    from tsukibot_pump.discovery.wallet_features import WalletFeatures

    feat = WalletFeatures(wallet="BOT")
    feat.n_observed_buys = 20
    feat.sol_amounts = [0.1] * 20  # zero variance
    feat.first_seen_unix = 1_000_000
    feat.last_seen_unix = 1_086_400
    flagged, reason = is_likely_bot(feat, BotHeuristicsConfig())
    assert flagged
    assert "mechanical" in reason


def test_bot_does_not_flag_human_pattern() -> None:
    from tsukibot_pump.discovery.wallet_features import WalletFeatures

    feat = WalletFeatures(wallet="HUMAN")
    feat.n_observed_buys = 20
    feat.sol_amounts = [0.05, 0.2, 0.15, 1.0, 0.3, 0.1, 0.5, 0.25, 0.8, 0.4] * 2
    feat.same_slot_entries = 0
    feat.first_seen_unix = 1_000_000
    feat.last_seen_unix = 1_086_400  # 20 trades/day
    flagged, reason = is_likely_bot(feat, BotHeuristicsConfig())
    assert not flagged, reason


# ── poison heuristics ─────────────────────────────────────────────────────


def test_poison_own_funder_is_dev_of_winner() -> None:
    """The smoking-gun case: KOL wallet is funded by the dev of a winner."""
    winners = [
        TokenOutcome(
            mint=f"M{i}",
            dev_wallet="DEV_X",
            first_seen_unix=1_000_000,
            first_price_sol_per_token=0.001,
            peak_price_sol_per_token=0.005,
            peak_multiple=5.0,
            last_real_sol_in_curve=2.0,
            label=TokenOutcomeLabel.WINNER,
        )
        for i in range(3)
    ]
    flagged, reason = is_likely_poison_wallet(
        "FAKE_KOL",
        winners,
        config=PoisonHeuristicsConfig(),
        funder_lookup={"FAKE_KOL": "DEV_X"},
    )
    assert flagged
    assert "honey" in reason


def test_poison_shared_funder_concentration() -> None:
    """Wins all come from tokens whose devs share a common funder."""
    winners = [
        TokenOutcome(
            mint=f"M{i}",
            dev_wallet=f"DEV_{i}",
            first_seen_unix=1_000_000,
            first_price_sol_per_token=0.001,
            peak_price_sol_per_token=0.005,
            peak_multiple=5.0,
            last_real_sol_in_curve=2.0,
            label=TokenOutcomeLabel.WINNER,
        )
        for i in range(4)
    ]
    flagged, reason = is_likely_poison_wallet(
        "FAKE_KOL",
        winners,
        config=PoisonHeuristicsConfig(max_shared_funder_fraction_of_wins=0.5),
        funder_lookup={
            "DEV_0": "UPSTREAM_A",
            "DEV_1": "UPSTREAM_A",
            "DEV_2": "UPSTREAM_A",
            "DEV_3": "UPSTREAM_B",
        },
    )
    assert flagged
    assert "coordinated" in reason


def test_poison_dump_bus_destination() -> None:
    """Most sells funnel to a single dump-bus address."""
    winners = [
        TokenOutcome(
            mint=f"M{i}",
            dev_wallet=f"DEV_{i}",
            first_seen_unix=1_000_000,
            first_price_sol_per_token=0.001,
            peak_price_sol_per_token=0.005,
            peak_multiple=5.0,
            last_real_sol_in_curve=2.0,
            label=TokenOutcomeLabel.WINNER,
        )
        for i in range(3)
    ]
    flagged, reason = is_likely_poison_wallet(
        "FAKE_KOL",
        winners,
        config=PoisonHeuristicsConfig(max_dump_bus_destination_fraction=0.7),
        sell_destination_lookup={
            "FAKE_KOL": ["BUS", "BUS", "BUS", "BUS", "OTHER"]  # 80% to BUS
        },
    )
    assert flagged
    assert "dump-bus" in reason


def test_poison_skips_check_with_too_few_trades() -> None:
    winners = [
        TokenOutcome(
            mint="M0",
            dev_wallet="DEV",
            first_seen_unix=1_000_000,
            first_price_sol_per_token=0.001,
            peak_price_sol_per_token=0.005,
            peak_multiple=5.0,
            last_real_sol_in_curve=2.0,
            label=TokenOutcomeLabel.WINNER,
        )
    ]
    flagged, _ = is_likely_poison_wallet(
        "FAKE_KOL",
        winners,
        config=PoisonHeuristicsConfig(min_positive_trades_for_check=3),
        funder_lookup={"FAKE_KOL": "DEV"},
    )
    assert not flagged  # only 1 positive trade → not enough data to call it


# ── end-to-end ────────────────────────────────────────────────────────────


def _build_synthetic_firehose() -> list[PumpEvent]:
    """Mix of two winners and two duds with three wallets:

    * HUMAN_KOL: bought both winners + one dud → 67% hit, geometric ROI ≈ 4x.
    * SNIPER_BOT: same-slot on every token (always slot=create_slot).
    * MEDIOCRE: bought only duds → 0% hit.
    """
    e: list[PumpEvent] = []
    now = 1_000_000

    # MINT_1 → winner (5x)
    e.append(_create("MINT_1", dev="DEV_1", slot=100, ts=now))
    e.append(
        _buy("MINT_1", "SNIPER_BOT", slot=100, ts=now + 1, sol=1.0, tokens=1000.0, dev="DEV_1")
    )
    e.append(
        _buy("MINT_1", "HUMAN_KOL", slot=110, ts=now + 30, sol=1.0, tokens=1000.0, dev="DEV_1")
    )
    e.append(_buy("MINT_1", "RANDO", slot=200, ts=now + 600, sol=5.0, tokens=1000.0, dev="DEV_1"))

    # MINT_2 → winner (4x)
    e.append(_create("MINT_2", dev="DEV_2", slot=300, ts=now + 1000))
    e.append(
        _buy("MINT_2", "SNIPER_BOT", slot=300, ts=now + 1001, sol=1.0, tokens=1000.0, dev="DEV_2")
    )
    e.append(
        _buy("MINT_2", "HUMAN_KOL", slot=320, ts=now + 1050, sol=2.0, tokens=1000.0, dev="DEV_2")
    )
    e.append(_buy("MINT_2", "RANDO", slot=400, ts=now + 1500, sol=4.0, tokens=1000.0, dev="DEV_2"))

    # MINT_3 → dead
    e.append(_create("MINT_3", dev="DEV_3", slot=500, ts=now + 2000))
    e.append(
        _buy("MINT_3", "SNIPER_BOT", slot=500, ts=now + 2001, sol=1.0, tokens=1000.0, dev="DEV_3")
    )
    e.append(
        _buy("MINT_3", "MEDIOCRE", slot=520, ts=now + 2100, sol=1.0, tokens=1000.0, dev="DEV_3")
    )
    e.append(
        _buy("MINT_3", "HUMAN_KOL", slot=530, ts=now + 2200, sol=1.0, tokens=1000.0, dev="DEV_3")
    )

    # MINT_4 → dead
    e.append(_create("MINT_4", dev="DEV_4", slot=600, ts=now + 3000))
    e.append(
        _buy("MINT_4", "SNIPER_BOT", slot=600, ts=now + 3001, sol=1.0, tokens=1000.0, dev="DEV_4")
    )
    e.append(
        _buy("MINT_4", "MEDIOCRE", slot=620, ts=now + 3100, sol=1.0, tokens=1000.0, dev="DEV_4")
    )

    # Add recent activity for HUMAN_KOL so the recency cut passes — many
    # small purchases over a few days.
    base = now + 4000
    for i in range(10):
        e.append(
            _buy(
                f"MINT_LATE_{i}",
                "HUMAN_KOL",
                slot=700 + i,
                ts=base + i * 3600,
                sol=0.1 + 0.05 * i,
                tokens=100.0,
            )
        )
    return e


def test_discover_kols_finds_human_kol_and_rejects_sniper() -> None:
    events = _build_synthetic_firehose()
    # Use the last observed timestamp + 1h as "now" so HUMAN_KOL's
    # activity is within the recency window.
    now_unix = max(e.block_time_unix or 0 for e in events if e.block_time_unix is not None) + 3600

    cfg = KolDiscoveryConfig(
        min_decided_trades=2,  # synthetic data is small
        observation_window_seconds=600,  # short enough that all real trades have outcomes
        top_n=10,
        bot_heuristics=BotHeuristicsConfig(
            max_same_slot_entry_fraction=0.5,
            max_trades_per_day=200.0,
            min_sol_size_cv=0.05,  # be lenient so HUMAN_KOL isn't flagged
            min_samples_for_cv=5,
        ),
    )
    kols = discover_kols(events, config=cfg, now_unix=now_unix)
    wallets = [k.wallet for k in kols]
    assert "HUMAN_KOL" in wallets, f"expected HUMAN_KOL in {wallets}"
    assert "SNIPER_BOT" not in wallets, "sniper should be filtered out"


def test_discover_kols_rejects_abandoned_wallet() -> None:
    """A wallet whose last trade is older than the recency window must be cut."""
    events = _build_synthetic_firehose()
    # Choose now_unix far in the future → everybody is "abandoned".
    now_unix = 999_999_999  # > 2 weeks after any event
    cfg = KolDiscoveryConfig(
        min_decided_trades=2,
        observation_window_seconds=600,
        require_last_trade_within_seconds=86_400,  # 1 day
        top_n=10,
    )
    kols = discover_kols(events, config=cfg, now_unix=now_unix)
    assert kols == [], f"expected empty leaderboard (all abandoned), got {kols}"


def test_write_kol_csv_uses_filter_compatible_format(tmp_path: Path) -> None:
    events = _build_synthetic_firehose()
    now_unix = max(e.block_time_unix or 0 for e in events if e.block_time_unix is not None) + 3600
    cfg = KolDiscoveryConfig(
        min_decided_trades=2,
        observation_window_seconds=600,
        top_n=10,
        bot_heuristics=BotHeuristicsConfig(
            max_same_slot_entry_fraction=0.5,
            max_trades_per_day=200.0,
            min_sol_size_cv=0.05,
            min_samples_for_cv=5,
        ),
    )
    kols = discover_kols(events, config=cfg, now_unix=now_unix)
    out_path = tmp_path / "kols.csv"
    n = write_kol_csv(kols, out_path)
    assert n > 0
    text = out_path.read_text(encoding="utf-8")
    assert text.startswith("# Auto-generated")
    # The first non-comment line should be parseable as wallet,label,score —
    # the same shape `first_kol_touch._load` expects.
    rows = [line for line in text.splitlines() if line and not line.startswith("#")]
    assert rows, "expected at least one wallet row"
    sample = rows[0].split(",")
    assert len(sample) >= 3
    assert sample[0] in {k.wallet for k in kols if not k.rejected}
    float(sample[2])  # score column must parse


def test_discover_kols_compatible_with_existing_filter(tmp_path: Path) -> None:
    """The CSV produced must be loadable by FirstKolTouch._load without error."""
    from tsukibot_pump.filters.first_kol_touch import FirstKolTouch

    events = _build_synthetic_firehose()
    now_unix = max(e.block_time_unix or 0 for e in events if e.block_time_unix is not None) + 3600
    cfg = KolDiscoveryConfig(
        min_decided_trades=2,
        observation_window_seconds=600,
        top_n=10,
        bot_heuristics=BotHeuristicsConfig(
            max_same_slot_entry_fraction=0.5,
            max_trades_per_day=200.0,
            min_sol_size_cv=0.05,
            min_samples_for_cv=5,
        ),
    )
    kols = discover_kols(events, config=cfg, now_unix=now_unix)
    csv_path = tmp_path / "private_kol_list.csv"
    write_kol_csv(kols, csv_path)

    loaded = FirstKolTouch._load(csv_path)
    assert loaded  # at least one entry, no exceptions
    for wallet, entry in loaded.items():
        assert entry.wallet == wallet
        assert isinstance(entry.score, float)


# ── CLI ───────────────────────────────────────────────────────────────────


def _events_to_jsonl(events: list[PumpEvent], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            row = {
                "kind": ev.kind.value,
                "signature": ev.signature,
                "slot": ev.slot,
                "block_time_unix": ev.block_time_unix,
                "mint": ev.mint,
                "dev_wallet": ev.dev_wallet,
                "actor_wallet": ev.actor_wallet,
                "sol_amount": ev.sol_amount,
                "token_amount": ev.token_amount,
            }
            f.write(json.dumps(row) + "\n")


def _build_large_synthetic_firehose() -> list[PumpEvent]:
    """Bigger fixture so HUMAN_KOL clears default ``min_decided_trades=5``.

    Events are spread one per day so the trades-per-day rate is realistic
    (~1/day) and HUMAN_KOL passes the bot-frequency check.
    """
    e: list[PumpEvent] = []
    now = 1_000_000
    day = 86_400
    for i in range(8):
        mint = f"WIN_{i}"
        slot = 100 + 50 * i
        ts = now + day * i
        e.append(_create(mint, dev=f"DEV_W{i}", slot=slot, ts=ts))
        e.append(
            _buy(mint, "SNIPER_BOT", slot=slot, ts=ts + 1, sol=1.0, tokens=1000.0, dev=f"DEV_W{i}")
        )
        e.append(
            _buy(
                mint,
                "HUMAN_KOL",
                slot=slot + 10,
                ts=ts + 30,
                sol=0.5 + 0.1 * i,
                tokens=1000.0,
                dev=f"DEV_W{i}",
            )
        )
        # Big later buyer to push price up → WINNER.
        e.append(
            _buy(mint, "RANDO", slot=slot + 30, ts=ts + 90, sol=5.0, tokens=1000.0, dev=f"DEV_W{i}")
        )
    for i in range(4):
        mint = f"DUD_{i}"
        slot = 1000 + 50 * i
        ts = now + day * (8 + i)
        e.append(_create(mint, dev=f"DEV_D{i}", slot=slot, ts=ts))
        e.append(
            _buy(mint, "SNIPER_BOT", slot=slot, ts=ts + 1, sol=1.0, tokens=1000.0, dev=f"DEV_D{i}")
        )
        e.append(
            _buy(
                mint,
                "MEDIOCRE",
                slot=slot + 10,
                ts=ts + 30,
                sol=0.5,
                tokens=1000.0,
                dev=f"DEV_D{i}",
            )
        )
    return e


def test_discovery_cli_end_to_end(tmp_path: Path) -> None:
    from tsukibot_pump.discovery.__main__ import main as discovery_main

    events = _build_large_synthetic_firehose()
    last_ts = max(e.block_time_unix or 0 for e in events if e.block_time_unix is not None)
    # `now` is well past the observation window so every event has a decided outcome.
    now_unix = last_ts + 25 * 3600
    events_path = tmp_path / "firehose.jsonl"
    out_path = tmp_path / "kols.csv"
    _events_to_jsonl(events, events_path)

    rc = discovery_main(
        [
            "--events",
            str(events_path),
            "--out",
            str(out_path),
            "--now-unix",
            str(now_unix),
            "--top-n",
            "10",
            "--log-level",
            "ERROR",
        ]
    )
    assert rc == 0, "CLI should succeed with the larger synthetic fixture"
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "HUMAN_KOL" in text
    assert "SNIPER_BOT" not in text


def test_discovery_cli_handles_missing_events_file(tmp_path: Path) -> None:
    from tsukibot_pump.discovery.__main__ import main as discovery_main

    rc = discovery_main(
        [
            "--events",
            str(tmp_path / "does_not_exist.jsonl"),
            "--out",
            str(tmp_path / "out.csv"),
            "--log-level",
            "ERROR",
        ]
    )
    assert rc == 2
