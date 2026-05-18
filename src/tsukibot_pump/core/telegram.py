"""Telegram bot — async, no SDK, identical pattern to tsuki-edge-bot.

Only sends to / accepts commands from one chat (`telegram_chat_id`). Long-
polls for /killswitch and /status. All errors are caught so a telegram
hiccup never crashes the trading loop.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)


class TelegramClient:
    """One bot, one chat. All messages go to the configured chat."""

    BASE_URL = "https://api.telegram.org"

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        *,
        http_timeout_seconds: float = 10.0,
    ) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._timeout = http_timeout_seconds
        self._client: httpx.AsyncClient | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._update_offset: int = 0
        self._command_handlers: dict[str, Callable[[str], Awaitable[None]]] = {}

    @property
    def configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    async def __aenter__(self) -> TelegramClient:
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self._stop_event.set()
        if self._poll_task is not None:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _url(self, method: str) -> str:
        return f"{self.BASE_URL}/bot{self.bot_token}/{method}"

    async def send_message(self, text: str, *, silent: bool = False) -> bool:
        """Send a message. Returns True on success. Never raises."""
        if not self.configured or self._client is None:
            return False
        try:
            resp = await self._client.post(
                self._url("sendMessage"),
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_notification": silent,
                },
            )
            ok = resp.status_code == 200 and resp.json().get("ok", False)
            if not ok:
                logger.warning(
                    "telegram.send_failed",
                    status=resp.status_code,
                    body=resp.text[:500],
                )
            return ok
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("telegram.send_exception", err=str(exc))
            return False

    def register_command(
        self,
        command: str,
        handler: Callable[[str], Awaitable[None]],
    ) -> None:
        self._command_handlers[command.lstrip("/")] = handler

    def start_polling(self) -> None:
        """Begin long-polling for commands in the background."""
        if not self.configured or self._client is None:
            return
        if self._poll_task is not None:
            return
        self._poll_task = asyncio.create_task(self._poll_loop(), name="telegram.poll")

    async def _poll_loop(self) -> None:
        assert self._client is not None
        while not self._stop_event.is_set():
            try:
                resp = await self._client.get(
                    self._url("getUpdates"),
                    params={
                        "offset": self._update_offset,
                        "timeout": 25,
                    },
                    timeout=30,
                )
                data: dict[str, Any] = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("telegram.poll_exception", err=str(exc))
                await asyncio.sleep(5)
                continue
            if not data.get("ok"):
                await asyncio.sleep(5)
                continue
            for update in data.get("result", []):
                self._update_offset = int(update["update_id"]) + 1
                msg = update.get("message") or update.get("channel_post")
                if not msg:
                    continue
                if str(msg.get("chat", {}).get("id")) != self.chat_id:
                    continue
                text: str = msg.get("text", "")
                if not text.startswith("/"):
                    continue
                parts = text.split(maxsplit=1)
                command = parts[0][1:].split("@")[0]
                rest = parts[1] if len(parts) > 1 else ""
                handler = self._command_handlers.get(command)
                if handler is None:
                    await self.send_message(f"unknown command: /{command}")
                    continue
                try:
                    await handler(rest)
                except Exception as exc:
                    logger.error("telegram.handler_error", command=command, err=str(exc))
                    await self.send_message(f"handler /{command} failed: {exc}")
