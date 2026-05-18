"""Structured logging via structlog with secret redaction.

Identical pattern to tsuki-edge-bot. We expand the redaction key list to
cover Solana / pump-bot specific names (hot wallet secret, RPC tokens).
"""

from __future__ import annotations

import contextlib
import logging
import logging.handlers
import sys
from pathlib import Path

import structlog
from structlog.types import EventDict, Processor

_SENSITIVE_KEY_FRAGMENTS = (
    "private_key",
    "secret",
    "api_secret",
    "passphrase",
    "telegram_bot_token",
    "solana_hot_wallet_secret",
    "solana_grpc_token",
    "birdeye_api_key",
    "dexscreener_api_key",
    "authorization",
    "keypair",
)


def _redact_secrets(_logger: object, _name: str, event_dict: EventDict) -> EventDict:
    """Best-effort redaction of obviously-sensitive keys.

    Defense in depth — production discipline is to never pass keys into a log
    call. This catches accidental slips.
    """
    for key in list(event_dict.keys()):
        lowered = key.lower()
        if any(s in lowered for s in _SENSITIVE_KEY_FRAGMENTS):
            event_dict[key] = "***redacted***"
    return event_dict


def configure_logging(
    log_level: str = "INFO",
    state_dir: Path | None = None,
    json_console: bool = False,
) -> None:
    """Configure structlog + stdlib logging.

    Args:
        log_level: stdlib log level name.
        state_dir: if given, a rotating log is written under it.
        json_console: emit JSON to stderr instead of pretty console renderer.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _redact_secrets,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
            if json_console
            else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
        # Close file handles so repeated re-configuration (e.g. across tests)
        # doesn't leak open fds; harmless for stream handlers.
        with contextlib.suppress(Exception):
            h.close()

    if state_dir is not None:
        state_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            state_dir / "tsuki-pump.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(file_handler)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a bound structlog logger."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
