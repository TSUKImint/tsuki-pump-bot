"""KillSwitch tests."""

from __future__ import annotations

import asyncio

import pytest

from tsukibot_pump.core.killswitch import KillSwitch


async def test_trip_sets_state() -> None:
    ks = KillSwitch()
    assert not ks.tripped
    ks.trip("test reason")
    assert ks.tripped
    assert ks.reason == "test reason"


async def test_trip_is_idempotent() -> None:
    ks = KillSwitch()
    ks.trip("first")
    ks.trip("second")
    assert ks.reason == "first"


async def test_wait_for_trip_returns_reason() -> None:
    ks = KillSwitch()
    task = asyncio.create_task(ks.wait_for_trip())
    await asyncio.sleep(0)
    ks.trip("from elsewhere")
    reason = await asyncio.wait_for(task, timeout=1)
    assert reason == "from elsewhere"


async def test_wait_for_trip_returns_immediately_if_already_tripped() -> None:
    ks = KillSwitch()
    ks.trip("early")
    reason = await asyncio.wait_for(ks.wait_for_trip(), timeout=0.1)
    assert reason == "early"


async def test_state_snapshot() -> None:
    ks = KillSwitch()
    snap = ks.state()
    assert snap.tripped is False
    ks.trip("snap")
    assert ks.state().tripped is True


# Ensure these are run inside the asyncio_mode = "auto" fixture
@pytest.mark.parametrize("reason", ["", "panic", "telegram /killswitch"])
async def test_trip_with_various_reasons(reason: str) -> None:
    ks = KillSwitch()
    ks.trip(reason)
    assert ks.reason == reason
