"""Tests for the performance reporter (PR #B).

Two layers:

* :mod:`tsukibot_pump.reporter.report` is pure functional — exercised
  with synthetic event / position dicts.
* :mod:`tsukibot_pump.reporter.__main__` is a thin asyncio wrapper —
  exercised with a real :class:`EventStore` round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tsukibot_pump.core.event_store import EventStore
from tsukibot_pump.reporter import (
    build_report,
    format_report_text,
)

# ── helpers ───────────────────────────────────────────────────────────────


def _event(
    kind: str,
    *,
    summary: str = "",
    payload: dict[str, Any] | None = None,
    mint: str | None = None,
    id_: int = 1,
) -> dict[str, Any]:
    return {
        "id": id_,
        "ts_utc": "2026-01-01T00:00:00+00:00",
        "kind": kind,
        "mint": mint,
        "dev_wallet": None,
        "severity": "info",
        "summary": summary,
        "payload_json": json.dumps(payload or {}),
    }


def _position(
    *,
    realized_pnl_sol: float | None,
    status: str = "closed",
    score_at_entry: float | None = 70.0,
    mint: str = "MINT_X",
    id_: int = 1,
) -> dict[str, Any]:
    return {
        "id": id_,
        "mint": mint,
        "dev_wallet": None,
        "side": "BUY",
        "entry_units": 1000.0,
        "entry_price_sol": 0.001,
        "entry_notional_sol": 1.0,
        "opened_at_utc": "2026-01-01T00:00:00+00:00",
        "closed_at_utc": "2026-01-01T01:00:00+00:00",
        "realized_pnl_sol": realized_pnl_sol,
        "status": status,
        "paper": 1,
        "score_at_entry": score_at_entry,
        "payload_json": "{}",
    }


# ── unit: build_report ────────────────────────────────────────────────────


def test_event_counts_aggregate_by_kind() -> None:
    events = [
        _event("paper.buy"),
        _event("paper.buy"),
        _event("paper.sell"),
        _event("risk.reject", summary="refused buy for ABC: spend rate"),
    ]
    report = build_report(events=events, positions=[])
    assert report.event_counts.total == 4
    assert report.event_counts.by_kind["paper.buy"] == 2
    assert report.event_counts.by_kind["paper.sell"] == 1
    assert report.event_counts.by_kind["risk.reject"] == 1


def test_reject_reasons_parsed_from_summary() -> None:
    events = [
        _event("risk.reject", summary="refused buy for ABC: spend rate"),
        _event("risk.reject", summary="refused buy for DEF: spend rate"),
        _event("risk.reject", summary="refused buy for XYZ: max concurrent open"),
    ]
    report = build_report(events=events, positions=[])
    assert report.reject_reasons == {"spend rate": 2, "max concurrent open": 1}


def test_paper_failures_counted_separately() -> None:
    events = [
        _event("paper.buy_failed"),
        _event("paper.buy_failed"),
        _event("paper.sell_failed"),
    ]
    report = build_report(events=events, positions=[])
    assert report.paper_fail_buy == 2
    assert report.paper_fail_sell == 1


def test_fee_breakdown_sums_diagnostic_fields() -> None:
    events = [
        _event(
            "paper.buy",
            payload={
                "fill": {
                    "pump_fee_sol": 0.01,
                    "priority_fee_sol": 0.001,
                    "drift_sol_absorbed": 0.05,
                    "latency_ms": 1500.0,
                }
            },
        ),
        _event(
            "paper.sell",
            payload={
                "fill": {
                    "pump_fee_sol": 0.02,
                    "priority_fee_sol": 0.002,
                    "drift_sol_absorbed": 0.10,
                    "latency_ms": 2300.0,
                }
            },
        ),
    ]
    report = build_report(events=events, positions=[])
    assert report.fees.fills_observed == 2
    assert report.fees.pump_fee_sol_total == pytest.approx(0.03)
    assert report.fees.priority_fee_sol_total == pytest.approx(0.003)
    assert report.fees.drift_sol_absorbed_total == pytest.approx(0.15)
    assert report.fees.total_friction_sol == pytest.approx(0.183)


def test_fee_breakdown_handles_missing_diagnostic_fields() -> None:
    # A fill from a non-realism executor has no `fill` key. The reporter
    # must not crash and must contribute zero to the totals.
    events = [
        _event("paper.buy", payload={"position_id": 7}),
        _event("paper.sell", payload={"pnl_sol": 0.05}),
    ]
    report = build_report(events=events, positions=[])
    assert report.fees.fills_observed == 0
    assert report.fees.total_friction_sol == 0.0


def test_latency_percentiles_are_correctly_ordered() -> None:
    events = [
        _event(
            "paper.buy",
            payload={
                "fill": {
                    "latency_ms": ms,
                    "pump_fee_sol": 0,
                    "priority_fee_sol": 0,
                    "drift_sol_absorbed": 0,
                }
            },
        )
        for ms in [100, 500, 1000, 2000, 5000, 10000]
    ]
    report = build_report(events=events, positions=[])
    assert report.latency is not None
    assert (
        report.latency.p50_ms
        < report.latency.p90_ms
        < report.latency.p99_ms
        <= report.latency.max_ms
    )
    assert report.latency.samples == 6


def test_overall_pnl_only_counts_closed_positions() -> None:
    positions = [
        _position(realized_pnl_sol=0.5, status="closed", id_=1),
        _position(realized_pnl_sol=-0.2, status="closed", id_=2),
        _position(realized_pnl_sol=None, status="open", id_=3),  # open, ignored
    ]
    report = build_report(events=[], positions=positions)
    assert report.overall.n_positions == 2
    assert report.overall.n_winners == 1
    assert report.overall.win_rate == pytest.approx(0.5)
    assert report.overall.total_pnl_sol == pytest.approx(0.3)


def test_score_buckets_separate_high_vs_low() -> None:
    positions = [
        _position(realized_pnl_sol=-0.1, score_at_entry=30.0, id_=1),  # <40
        _position(realized_pnl_sol=-0.05, score_at_entry=50.0, id_=2),  # 40-60
        _position(realized_pnl_sol=0.05, score_at_entry=70.0, id_=3),  # 60-80
        _position(realized_pnl_sol=0.4, score_at_entry=85.0, id_=4),  # 80+
        _position(realized_pnl_sol=0.6, score_at_entry=95.0, id_=5),  # 80+
    ]
    report = build_report(events=[], positions=positions)
    by_label = {b.label: b for b in report.score_buckets}

    assert by_label["0-40"].stats.n_positions == 1
    assert by_label["0-40"].stats.total_pnl_sol == pytest.approx(-0.1)
    assert by_label["80+"].stats.n_positions == 2
    assert by_label["80+"].stats.win_rate == pytest.approx(1.0)
    assert by_label["80+"].stats.total_pnl_sol == pytest.approx(1.0)


def test_exit_reasons_grouped_and_sorted_by_total_pnl() -> None:
    events = [
        _event("paper.sell", payload={"reason": "stop", "pnl_sol": -0.1}),
        _event("paper.sell", payload={"reason": "stop", "pnl_sol": -0.2}),
        _event("paper.sell", payload={"reason": "take_profit_1", "pnl_sol": 0.5}),
        _event("paper.sell", payload={"reason": "take_profit_2", "pnl_sol": 0.3}),
    ]
    report = build_report(events=events, positions=[])
    # Sorted by total_pnl descending.
    by_reason = {r.reason: r for r in report.exit_reasons}
    assert by_reason["take_profit_1"].total_pnl_sol == pytest.approx(0.5)
    assert by_reason["stop"].n_exits == 2
    assert by_reason["stop"].mean_pnl_sol == pytest.approx(-0.15)


def test_sharpe_undefined_when_only_one_position() -> None:
    positions = [_position(realized_pnl_sol=0.5, id_=1)]
    report = build_report(events=[], positions=positions)
    assert report.overall.pnl_sharpe is None


def test_sharpe_undefined_when_all_pnl_equal() -> None:
    positions = [_position(realized_pnl_sol=0.5, id_=i) for i in range(3)]
    report = build_report(events=[], positions=positions)
    # Stdev is zero → sharpe undefined, not infinite.
    assert report.overall.pnl_sharpe is None


def test_format_report_text_contains_all_sections() -> None:
    events = [
        _event(
            "paper.buy",
            payload={
                "fill": {
                    "pump_fee_sol": 0.01,
                    "priority_fee_sol": 0.001,
                    "drift_sol_absorbed": 0.02,
                    "latency_ms": 1500,
                }
            },
        ),
        _event(
            "paper.sell",
            payload={
                "fill": {
                    "pump_fee_sol": 0.01,
                    "priority_fee_sol": 0.001,
                    "drift_sol_absorbed": 0.02,
                    "latency_ms": 1700,
                },
                "reason": "stop",
                "pnl_sol": -0.1,
            },
        ),
        _event("paper.buy_failed"),
        _event("risk.reject", summary="refused buy for ABC: spend rate"),
    ]
    positions = [_position(realized_pnl_sol=-0.1, score_at_entry=85.0)]
    report = build_report(events=events, positions=positions)
    text = format_report_text(report)

    for marker in (
        "tsuki-pump performance report",
        "[events]",
        "[reliability]",
        "[closed positions — overall]",
        "[closed positions — by score_at_entry bucket]",
        "[exits — by reason]",
        "[paper-realism friction]",
        "[fill latency]",
    ):
        assert marker in text, f"missing section: {marker!r}\n{text}"


def test_format_report_text_handles_zero_data() -> None:
    """No tracebacks when there are zero events and zero positions."""
    report = build_report(events=[], positions=[])
    text = format_report_text(report)
    assert "(no events recorded)" in text
    assert "(no closed positions)" in text


def test_payload_already_parsed_as_dict_is_accepted() -> None:
    """Real callers can pass a dict instead of a JSON string."""
    event = {
        "id": 1,
        "ts_utc": "2026-01-01T00:00:00+00:00",
        "kind": "paper.buy",
        "mint": None,
        "dev_wallet": None,
        "severity": "info",
        "summary": "",
        "payload": {
            "fill": {
                "pump_fee_sol": 0.01,
                "priority_fee_sol": 0.001,
                "drift_sol_absorbed": 0.02,
                "latency_ms": 1500,
            }
        },
    }
    report = build_report(events=[event], positions=[])
    assert report.fees.pump_fee_sol_total == pytest.approx(0.01)


def test_score_buckets_skips_positions_without_score() -> None:
    positions = [
        _position(realized_pnl_sol=0.1, score_at_entry=None, id_=1),
        _position(realized_pnl_sol=0.2, score_at_entry=85.0, id_=2),
    ]
    report = build_report(events=[], positions=positions)
    # Both contribute to "overall" — but only the scored one goes into a bucket.
    assert report.overall.n_positions == 2
    total_bucketed = sum(b.stats.n_positions for b in report.score_buckets)
    assert total_bucketed == 1


# ── integration: CLI + real EventStore ────────────────────────────────────


@pytest.mark.asyncio
async def test_reporter_round_trip_through_event_store(tmp_path: Path) -> None:
    """Write events + positions, then read them back via the CLI loader."""
    from tsukibot_pump.reporter.__main__ import _load_rows

    db = tmp_path / "store.sqlite"
    audit = tmp_path / "audit.jsonl"
    async with EventStore(db, audit) as store:
        # One paper buy + one losing paper sell.
        await store.record_event(
            "paper.buy",
            "test buy",
            payload={
                "fill": {
                    "pump_fee_sol": 0.01,
                    "priority_fee_sol": 0.001,
                    "drift_sol_absorbed": 0.02,
                    "latency_ms": 1500,
                }
            },
        )
        await store.record_event(
            "paper.sell",
            "test sell",
            payload={
                "fill": {
                    "pump_fee_sol": 0.01,
                    "priority_fee_sol": 0.001,
                    "drift_sol_absorbed": 0.02,
                    "latency_ms": 1800,
                },
                "reason": "stop",
                "pnl_sol": -0.05,
            },
        )
        pos_id = await store.open_position(
            mint="TEST_MINT",
            dev_wallet=None,
            entry_units=1000.0,
            entry_price_sol=0.001,
            entry_notional_sol=1.0,
            paper=True,
            score_at_entry=85.0,
        )
        await store.close_position(pos_id, realized_pnl_sol=-0.05)

    events, positions = await _load_rows(db, max_events=10_000)
    report = build_report(events=events, positions=positions)
    assert report.overall.n_positions == 1
    assert report.overall.total_pnl_sol == pytest.approx(-0.05)
    assert report.fees.pump_fee_sol_total == pytest.approx(0.02)


def test_reporter_cli_missing_db_returns_two(tmp_path: Path) -> None:
    from tsukibot_pump.reporter.__main__ import main as reporter_main

    rc = reporter_main(
        [
            "--db",
            str(tmp_path / "does_not_exist.sqlite"),
            "--log-level",
            "ERROR",
        ]
    )
    assert rc == 2


def test_reporter_cli_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """End-to-end: write to a real DB, run the CLI in --json mode, parse result."""
    import asyncio

    from tsukibot_pump.reporter.__main__ import main as reporter_main

    db = tmp_path / "store.sqlite"
    audit = tmp_path / "audit.jsonl"

    async def _setup() -> None:
        async with EventStore(db, audit) as store:
            await store.record_event("paper.buy", "buy")
            pos_id = await store.open_position(
                mint="MINT",
                dev_wallet=None,
                entry_units=1.0,
                entry_price_sol=0.001,
                entry_notional_sol=0.001,
                paper=True,
                score_at_entry=75.0,
            )
            await store.close_position(pos_id, realized_pnl_sol=0.02)

    asyncio.run(_setup())

    rc = reporter_main(
        [
            "--db",
            str(db),
            "--json",
            "--log-level",
            "ERROR",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["overall"]["n_positions"] == 1
    assert parsed["overall"]["total_pnl_sol"] == pytest.approx(0.02)
