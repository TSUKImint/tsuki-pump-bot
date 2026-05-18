"""Global kill switch — identical pattern to tsuki-edge-bot.

When tripped:
  - all strategies refuse to open new positions
  - the orchestrator initiates cancel-all on the executor
  - the orchestrator cancels all running asyncio tasks
  - Telegram fires an alert (if configured)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class KillSwitchState:
    tripped: bool
    reason: str = ""


class KillSwitch:
    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: str = ""

    @property
    def tripped(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def state(self) -> KillSwitchState:
        return KillSwitchState(tripped=self.tripped, reason=self._reason)

    def trip(self, reason: str) -> None:
        if self._event.is_set():
            return
        self._reason = reason
        self._event.set()

    async def wait_for_trip(self) -> str:
        await self._event.wait()
        return self._reason
