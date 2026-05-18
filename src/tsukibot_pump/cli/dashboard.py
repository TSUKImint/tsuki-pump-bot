"""Real-time CLI dashboard for the pump-bot.

Four panels:
  * header — mode, kill-switch, scout stats, uptime
  * top scoring tokens — last cycle's top 8 by composite score
  * open positions — paper PnL, peak/current price, ladder progress
  * recent events — last 12 from the event store (in-memory ring buffer)

Reads from in-memory state passed in by the orchestrator. Never writes.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.killswitch import KillSwitch
from ..core.risk import RiskEngine
from ..execution.position_monitor import Position
from ..models import TokenState
from ..scout.pump_scout import ScoutStats


class _ScoutStatsProto(Protocol):
    """Structural type for any scout-stats object the dashboard renders.

    Lets the dashboard accept either `ScoutStats` (HTTP-poll) or
    `HeliusScoutStats` (WS) without importing both.
    """

    signatures_seen: int
    transactions_parsed: int
    create_events: int
    buy_events: int
    sell_events: int
    rpc_errors: int
    last_signature: str


class DashboardState:
    """Shared, mutable view exposed to the dashboard by the orchestrator."""

    def __init__(self, *, max_recent_events: int = 50) -> None:
        self.scout_stats: _ScoutStatsProto = ScoutStats()
        self.top_tokens: list[TokenState] = []
        self.open_positions: dict[str, Position] = {}
        self.current_prices: dict[str, float] = {}
        self.recent_events: deque[dict[str, Any]] = deque(maxlen=max_recent_events)
        self.tokens_scored_total: int = 0
        self.tokens_rejected_total: int = 0


class Dashboard:
    def __init__(
        self,
        *,
        mode: str,
        bankroll_sol: float,
        kill: KillSwitch,
        risk: RiskEngine,
        state: DashboardState,
        refresh_hz: int,
        get_started_at: Callable[[], datetime],
    ) -> None:
        self.mode = mode
        self.bankroll_sol = bankroll_sol
        self.kill = kill
        self.risk = risk
        self.state = state
        self.refresh_hz = max(1, min(refresh_hz, 60))
        self._get_started_at = get_started_at
        self.console = Console(stderr=False)

    async def run_forever(self) -> None:
        with Live(
            self._render(),
            console=self.console,
            refresh_per_second=self.refresh_hz,
            screen=False,
            transient=False,
        ) as live:
            while True:
                live.update(self._render())
                await asyncio.sleep(1.0 / self.refresh_hz)

    def _render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._header(), name="header", size=5),
            Layout(self._top_tokens(), name="top", ratio=2),
            Layout(self._positions(), name="positions", ratio=2),
            Layout(self._events(), name="events", ratio=2),
        )
        return layout

    def _header(self) -> Panel:
        now = datetime.now(tz=UTC)
        uptime = now - self._get_started_at()
        kill_style = "bold red" if self.kill.tripped else "bold green"
        kill_text = f"KILLED ({self.kill.reason})" if self.kill.tripped else "running"
        mode_style = {
            "paper-mock": "bold cyan",
            "paper": "bold cyan",
            "devnet": "bold yellow",
            "mainnet": "bold red",
        }.get(self.mode, "bold white")
        stats = self.state.scout_stats
        header = Text()
        header.append("tsuki-pump-bot v0.2 ", style="bold magenta")
        header.append(f"[{self.mode.upper()}] ", style=mode_style)
        header.append(" | ", style="dim")
        header.append(kill_text, style=kill_style)
        header.append(f" | utc {now.strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
        header.append(f" | uptime {_fmt_td(uptime)}", style="dim")
        header.append(
            f" | bankroll {self.bankroll_sol:.3f} SOL "
            f"| realized {self.risk.pnl.realized_sol:+.3f} SOL "
            f"| unrealized {self.risk.pnl.unrealized_sol:+.3f} SOL",
            style="dim",
        )
        header.append(
            f"\nscout: sigs={stats.signatures_seen} txs={stats.transactions_parsed} "
            f"creates={stats.create_events} buys={stats.buy_events} sells={stats.sell_events} "
            f"errors={stats.rpc_errors}"
            f" | scored={self.state.tokens_scored_total}"
            f" rejected={self.state.tokens_rejected_total}",
            style="dim",
        )
        return Panel(header, border_style="magenta")

    def _top_tokens(self) -> Panel:
        table = Table(expand=True, show_header=True, header_style="bold cyan")
        table.add_column("mint", overflow="fold")
        table.add_column("symbol")
        table.add_column("sol-in-curve", justify="right")
        table.add_column("v sol/min", justify="right")
        table.add_column("buys/60s", justify="right")
        table.add_column("score", justify="right")
        table.add_column("notes", overflow="fold")

        tokens: Sequence[TokenState] = self.state.top_tokens[:8]
        if not tokens:
            table.add_row("[dim](nothing has scored yet)[/dim]", "", "", "", "", "", "")
        for tk in tokens:
            score_style = (
                "green"
                if tk.last_composite_score >= 60
                else "yellow"
                if tk.last_composite_score >= 40
                else "red"
            )
            reason = tk.last_decision_reason
            if tk.rejected and tk.reject_reason:
                reason = f"REJECT: {tk.reject_reason}"
            table.add_row(
                tk.mint[:14] + "…",
                (tk.symbol or "")[:10],
                f"{tk.sol_in_curve:.2f}",
                f"{tk.last_sol_velocity_sol_per_min:.2f}",
                str(tk.distinct_buyers_60s),
                f"[{score_style}]{tk.last_composite_score:.1f}[/{score_style}]",
                reason[:80],
            )
        return Panel(table, title="top tokens (this cycle)", border_style="cyan")

    def _positions(self) -> Panel:
        table = Table(expand=True, show_header=True, header_style="bold green")
        table.add_column("mint", overflow="fold")
        table.add_column("units", justify="right")
        table.add_column("entry $/tok", justify="right")
        table.add_column("now $/tok", justify="right")
        table.add_column("ROI", justify="right")
        table.add_column("realized SOL", justify="right")
        table.add_column("ladder")

        if not self.state.open_positions:
            table.add_row("[dim](no open positions)[/dim]", "", "", "", "", "", "")
        for mint, pos in self.state.open_positions.items():
            current = self.state.current_prices.get(mint, pos.entry_price_sol_per_token)
            roi = (
                (current - pos.entry_price_sol_per_token) / pos.entry_price_sol_per_token
                if pos.entry_price_sol_per_token
                else 0.0
            )
            style = "green" if roi > 0 else "red" if roi < 0 else "white"
            table.add_row(
                mint[:14] + "…",
                f"{pos.units_held:.4f}",
                f"{pos.entry_price_sol_per_token:.8f}",
                f"{current:.8f}",
                f"[{style}]{roi:+.2%}[/{style}]",
                _coloured_money_sol(pos.realized_pnl_sol),
                f"{len(pos.ladder_steps_done)} steps done",
            )
        return Panel(table, title="open paper positions", border_style="green")

    def _events(self) -> Panel:
        table = Table(expand=True, show_header=True, header_style="bold blue")
        table.add_column("ts", style="dim")
        table.add_column("kind", style="cyan")
        table.add_column("sev")
        table.add_column("summary", overflow="fold")
        events = list(self.state.recent_events)[-12:]
        events.reverse()
        if not events:
            table.add_row("", "", "", "[dim](no events yet)[/dim]")
        for e in events:
            sev = str(e.get("severity", "info"))
            sev_style = {"error": "red", "warning": "yellow", "info": "white"}.get(sev, "white")
            ts = str(e.get("ts_utc", ""))[11:19]
            table.add_row(
                ts,
                str(e.get("kind", "?")),
                f"[{sev_style}]{sev}[/{sev_style}]",
                str(e.get("summary", ""))[:200],
            )
        return Panel(Group(table), title="recent events", border_style="blue")


def _coloured_money_sol(amount: float) -> str:
    if amount > 0:
        return f"[green]+{amount:.4f}[/green]"
    if amount < 0:
        return f"[red]{amount:.4f}[/red]"
    return f"{amount:.4f}"


def _fmt_td(td: object) -> str:
    s = str(td).split(".", 1)[0]
    return s
