"""PositionMonitor exit-decision tests."""

from __future__ import annotations

from tsukibot_pump.config import ExitLadderStep, ExitsConfig
from tsukibot_pump.execution.position_monitor import Position, PositionMonitor


def _config() -> ExitsConfig:
    return ExitsConfig(
        ladder=[
            ExitLadderStep(roi=1.0, sell_fraction=0.5),
            ExitLadderStep(roi=3.0, sell_fraction=0.3),
        ],
        trailing_stop_pct=0.4,
        dev_drain_exit_fraction=0.1,
        hard_stop_loss_pct=0.5,
    )


def _position(units: float = 100.0, entry: float = 1.0) -> Position:
    return Position(
        mint="M",
        units_held=units,
        entry_units=units,
        entry_price_sol_per_token=entry,
        peak_price_sol_per_token=entry,
    )


def test_no_action_at_entry_price() -> None:
    mon = PositionMonitor(_config())
    actions = mon.evaluate(_position(), current_price=1.0)
    assert actions == []


def test_ladder_first_step_at_2x() -> None:
    mon = PositionMonitor(_config())
    pos = _position()
    actions = mon.evaluate(pos, current_price=2.0)
    assert len(actions) == 1
    assert actions[0].units_to_sell == 50.0
    assert "ladder@1.0x" in actions[0].reason
    assert 0 in pos.ladder_steps_done


def test_ladder_does_not_double_fire() -> None:
    mon = PositionMonitor(_config())
    pos = _position()
    mon.evaluate(pos, current_price=2.0)
    pos.units_held = 50.0
    # Price stays at 2x — same ladder step shouldn't fire again.
    actions = mon.evaluate(pos, current_price=2.0)
    assert actions == []


def test_ladder_second_step_at_4x() -> None:
    mon = PositionMonitor(_config())
    pos = _position()
    mon.evaluate(pos, current_price=2.0)
    pos.units_held = 50.0
    actions = mon.evaluate(pos, current_price=4.0)
    assert len(actions) == 1
    assert actions[0].units_to_sell == 30.0
    assert "ladder@3.0x" in actions[0].reason


def test_trailing_stop_only_after_all_ladder_done() -> None:
    mon = PositionMonitor(_config())
    pos = _position()
    # Climb through both ladders.
    mon.evaluate(pos, current_price=2.0)
    pos.units_held = 50.0
    mon.evaluate(pos, current_price=4.0)
    pos.units_held = 20.0
    # Now drop 40% from peak (4.0 → 2.4).
    actions = mon.evaluate(pos, current_price=2.4)
    assert any(a.reason == "trailing" for a in actions)
    trailing = next(a for a in actions if a.reason == "trailing")
    assert trailing.units_to_sell == 20.0


def test_hard_stop_loss_exits_full_position() -> None:
    mon = PositionMonitor(_config())
    pos = _position()
    actions = mon.evaluate(pos, current_price=0.4)  # -60%
    assert len(actions) == 1
    assert actions[0].units_to_sell == 100.0
    assert actions[0].reason == "hard_stop"


def test_dev_drain_exits_full_position() -> None:
    mon = PositionMonitor(_config())
    pos = _position()
    actions = mon.evaluate(pos, current_price=1.05, dev_drained_fraction=0.15)
    assert len(actions) == 1
    assert actions[0].reason == "dev_drain"


def test_peak_price_tracks_high() -> None:
    mon = PositionMonitor(_config())
    pos = _position()
    mon.evaluate(pos, current_price=1.5)
    assert pos.peak_price_sol_per_token == 1.5
    mon.evaluate(pos, current_price=1.2)
    assert pos.peak_price_sol_per_token == 1.5  # peak doesn't go down


def test_empty_position_yields_no_actions() -> None:
    mon = PositionMonitor(_config())
    pos = _position(units=0.0)
    assert mon.evaluate(pos, current_price=2.0) == []
