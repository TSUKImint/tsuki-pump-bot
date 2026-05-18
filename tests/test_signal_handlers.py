"""Signal handler installation — Windows fallback regression tests."""

from __future__ import annotations

import asyncio
import signal
from typing import Any

import pytest

from tsukibot_pump.__main__ import _install_signal_handlers, _make_signal_handler
from tsukibot_pump.core.killswitch import KillSwitch


def test_make_signal_handler_trips_kill_switch() -> None:
    ks = KillSwitch()
    handler = _make_signal_handler(signal.SIGINT, ks)
    handler()
    assert ks.tripped
    assert ks.reason == "signal SIGINT"


async def test_install_signal_handlers_unsupported_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If `add_signal_handler` raises NotImplementedError (Windows path),
    `_install_signal_handlers` must swallow it cleanly."""

    class _Loop:
        def add_signal_handler(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError

    ks = KillSwitch()
    fake_loop = _Loop()

    # No-op logger object — only `debug` is called.
    class _Logger:
        def debug(self, *args: Any, **kwargs: Any) -> None:
            pass

    # Should not raise.
    _install_signal_handlers(fake_loop, ks, _Logger())  # type: ignore[arg-type]
    assert not ks.tripped


async def test_install_signal_handlers_succeeds_on_real_loop() -> None:
    """On POSIX, real loops install handlers — should not raise."""
    ks = KillSwitch()
    loop = asyncio.get_running_loop()

    class _Logger:
        def debug(self, *args: Any, **kwargs: Any) -> None:
            pass

    try:
        _install_signal_handlers(loop, ks, _Logger())  # type: ignore[arg-type]
    except NotImplementedError:
        pytest.skip("platform does not support add_signal_handler")
    # We're on Linux/macOS — the install path should have completed silently.
    assert not ks.tripped
