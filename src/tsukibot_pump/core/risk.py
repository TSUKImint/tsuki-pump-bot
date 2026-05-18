"""Risk engine: PnL accounting + drawdown limits, denominated in SOL.

Mirrors the tsuki-edge-bot pattern but tracks SOL instead of USDC. Pure
arithmetic, single-threaded, must be called by the orchestrator before
opening any new position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime


@dataclass
class StrategyPnL:
    """PnL accounting denominated in SOL."""

    starting_bankroll_sol: float
    realized_sol: float = 0.0
    unrealized_sol: float = 0.0
    daily_realized_sol: float = 0.0
    last_reset_date: date = field(default_factory=lambda: datetime.now(tz=UTC).date())

    def total_pnl(self) -> float:
        return self.realized_sol + self.unrealized_sol

    def total_drawdown_fraction(self) -> float:
        if self.starting_bankroll_sol <= 0:
            return 0.0
        return -self.total_pnl() / self.starting_bankroll_sol

    def daily_drawdown_fraction(self) -> float:
        if self.starting_bankroll_sol <= 0:
            return 0.0
        return -self.daily_realized_sol / self.starting_bankroll_sol

    def maybe_reset_daily(self, now: datetime | None = None) -> None:
        now = now or datetime.now(tz=UTC)
        today = now.date()
        if today != self.last_reset_date:
            self.daily_realized_sol = 0.0
            self.last_reset_date = today


@dataclass
class RiskLimits:
    daily_drawdown_kill: float
    total_drawdown_kill: float
    single_token_cap_fraction: float
    max_open_positions: int
    hard_cap_per_trade_sol: float


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""


class RiskEngine:
    """Per-bot PnL bookkeeping + limit enforcement."""

    def __init__(self, starting_bankroll_sol: float, limits: RiskLimits) -> None:
        self.pnl = StrategyPnL(starting_bankroll_sol=starting_bankroll_sol)
        self.limits = limits
        self.open_notional_sol: float = 0.0
        self.open_positions: int = 0
        self._tripped: bool = False
        self._trip_reason: str = ""

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def trip_reason(self) -> str:
        return self._trip_reason

    def _evaluate_limits(self) -> None:
        if self._tripped:
            return
        self.pnl.maybe_reset_daily()
        if self.pnl.total_drawdown_fraction() >= self.limits.total_drawdown_kill:
            self._tripped = True
            self._trip_reason = (
                f"total drawdown {self.pnl.total_drawdown_fraction():.2%} "
                f">= limit {self.limits.total_drawdown_kill:.2%}"
            )
        elif self.pnl.daily_drawdown_fraction() >= self.limits.daily_drawdown_kill:
            self._tripped = True
            self._trip_reason = (
                f"daily drawdown {self.pnl.daily_drawdown_fraction():.2%} "
                f">= limit {self.limits.daily_drawdown_kill:.2%}"
            )

    def check_open_allowed(self, notional_sol: float) -> RiskDecision:
        self._evaluate_limits()
        if self._tripped:
            return RiskDecision(False, f"risk tripped: {self._trip_reason}")
        if notional_sol <= 0:
            return RiskDecision(False, "notional must be positive")
        if notional_sol > self.limits.hard_cap_per_trade_sol:
            return RiskDecision(
                False,
                f"notional {notional_sol:.4f} SOL exceeds hard cap "
                f"{self.limits.hard_cap_per_trade_sol:.4f} SOL",
            )
        cap = self.pnl.starting_bankroll_sol * self.limits.single_token_cap_fraction
        if notional_sol > cap:
            return RiskDecision(
                False,
                f"notional {notional_sol:.4f} SOL exceeds single-token cap {cap:.4f} SOL",
            )
        if self.open_positions >= self.limits.max_open_positions:
            return RiskDecision(
                False,
                f"open positions {self.open_positions} at limit {self.limits.max_open_positions}",
            )
        return RiskDecision(True)

    def record_open(self, notional_sol: float) -> None:
        self.open_notional_sol += notional_sol
        self.open_positions += 1

    def record_close(self, realized_pnl_sol: float, notional_freed_sol: float) -> None:
        self.pnl.realized_sol += realized_pnl_sol
        self.pnl.daily_realized_sol += realized_pnl_sol
        self.open_notional_sol = max(0.0, self.open_notional_sol - notional_freed_sol)
        self.open_positions = max(0, self.open_positions - 1)
        self._evaluate_limits()

    def update_unrealized(self, unrealized_sol: float) -> None:
        self.pnl.unrealized_sol = unrealized_sol
        self._evaluate_limits()

    def manual_trip(self, reason: str) -> None:
        self._tripped = True
        self._trip_reason = reason
