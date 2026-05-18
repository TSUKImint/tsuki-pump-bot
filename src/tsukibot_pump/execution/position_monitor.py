"""Position monitor — exit ladder, dev drain, trailing stop, hard stop.

Pure logic module: takes a `Position` + current price + dev-activity update
and returns 0..n `ExitAction` records describing which fractions to sell.
The orchestrator owns the IO (calling the executor + persisting events).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import ExitsConfig


@dataclass(slots=True)
class Position:
    mint: str
    units_held: float  # current holdings (after partial sells)
    entry_units: float  # original size
    entry_price_sol_per_token: float
    peak_price_sol_per_token: float = 0.0  # for trailing stop
    realized_pnl_sol: float = 0.0
    ladder_steps_done: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ExitAction:
    units_to_sell: float
    reason: str  # "ladder@2x" | "trailing" | "hard_stop" | "dev_drain"


class PositionMonitor:
    """Stateless exit decisioner."""

    def __init__(self, exits_config: ExitsConfig) -> None:
        self.config = exits_config

    def update_peak(self, position: Position, current_price: float) -> None:
        if current_price > position.peak_price_sol_per_token:
            position.peak_price_sol_per_token = current_price

    def evaluate(
        self,
        position: Position,
        current_price: float,
        *,
        dev_drained_fraction: float = 0.0,
    ) -> list[ExitAction]:
        actions: list[ExitAction] = []
        if position.units_held <= 0:
            return actions

        self.update_peak(position, current_price)
        entry = position.entry_price_sol_per_token
        if entry <= 0:
            return actions

        roi = (current_price - entry) / entry

        # 1) Hard stop loss → exit 100% immediately.
        if roi <= -self.config.hard_stop_loss_pct:
            actions.append(ExitAction(units_to_sell=position.units_held, reason="hard_stop"))
            return actions

        # 2) Dev drain → exit 100% immediately.
        if dev_drained_fraction >= self.config.dev_drain_exit_fraction:
            actions.append(ExitAction(units_to_sell=position.units_held, reason="dev_drain"))
            return actions

        # 3) Ladder steps — sell pre-set fractions at multiples of entry.
        for idx, step in enumerate(self.config.ladder):
            if idx in position.ladder_steps_done:
                continue
            if roi >= step.roi:
                # Sell `sell_fraction` of the ORIGINAL entry units, not
                # current holdings — so ladders compose predictably.
                units = step.sell_fraction * position.entry_units
                units = min(units, position.units_held)
                if units > 0:
                    actions.append(
                        ExitAction(units_to_sell=units, reason=f"ladder@{step.roi:.1f}x")
                    )
                position.ladder_steps_done.append(idx)

        # 4) Trailing stop — only on the residual (after all ladder steps).
        all_ladder_done = len(position.ladder_steps_done) == len(self.config.ladder)
        if all_ladder_done and position.units_held > 0 and position.peak_price_sol_per_token > 0:
            drawdown = (
                position.peak_price_sol_per_token - current_price
            ) / position.peak_price_sol_per_token
            if drawdown >= self.config.trailing_stop_pct:
                actions.append(ExitAction(units_to_sell=position.units_held, reason="trailing"))
        return actions
