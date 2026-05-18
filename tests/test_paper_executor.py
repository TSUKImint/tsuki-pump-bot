"""PaperExecutor tests."""

from __future__ import annotations

import pytest

from tsukibot_pump.execution.paper_executor import PaperExecutor
from tsukibot_pump.solana.bonding_curve import (
    DEFAULT_VIRTUAL_SOL_RESERVES,
    DEFAULT_VIRTUAL_TOKEN_RESERVES,
    BondingCurveState,
)


def _fresh_curve() -> BondingCurveState:
    return BondingCurveState(
        virtual_sol_reserves=DEFAULT_VIRTUAL_SOL_RESERVES,
        virtual_token_reserves=DEFAULT_VIRTUAL_TOKEN_RESERVES,
        real_sol_reserves=0,
        real_token_reserves=793_100_000 * 10**6,
        complete=False,
    )


def test_buy_applies_slippage_upward() -> None:
    no_slip = PaperExecutor(slippage_bps=0.0)
    with_slip = PaperExecutor(slippage_bps=100.0)  # 1%
    curve = _fresh_curve()
    f1 = no_slip.buy("M", 100_000, curve)
    f2 = with_slip.buy("M", 100_000, curve)
    assert f2.notional_sol > f1.notional_sol
    assert f2.fill_price_sol_per_token > f1.fill_price_sol_per_token
    assert f1.slippage_bps_applied == 0.0
    assert f2.slippage_bps_applied == 100.0


def test_sell_applies_slippage_downward() -> None:
    no_slip = PaperExecutor(slippage_bps=0.0)
    with_slip = PaperExecutor(slippage_bps=100.0)
    curve = BondingCurveState(
        virtual_sol_reserves=DEFAULT_VIRTUAL_SOL_RESERVES + 10 * 1_000_000_000,
        virtual_token_reserves=DEFAULT_VIRTUAL_TOKEN_RESERVES - 10_000_000_000,
        real_sol_reserves=10 * 1_000_000_000,
        real_token_reserves=783_100_000 * 10**6,
        complete=False,
    )
    f1 = no_slip.sell("M", 10_000, curve)
    f2 = with_slip.sell("M", 10_000, curve)
    assert f2.notional_sol < f1.notional_sol


def test_negative_slippage_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PaperExecutor(slippage_bps=-1.0)


def test_buy_zero_units_rejected() -> None:
    exe = PaperExecutor(slippage_bps=0.0)
    with pytest.raises(ValueError, match="positive"):
        exe.buy("M", 0.0, _fresh_curve())


def test_buy_more_than_curve_rejected() -> None:
    exe = PaperExecutor(slippage_bps=0.0)
    huge_curve = _fresh_curve()
    huge_units = (huge_curve.virtual_token_reserves / 10**6) - 1
    with pytest.raises(ValueError, match="insufficient depth"):
        # Asking for everything → curve math returns inf → executor rejects.
        exe.buy("M", huge_units * 2, huge_curve)


def test_paper_fill_marked_paper() -> None:
    exe = PaperExecutor(slippage_bps=50.0)
    fill = exe.buy("MINT_X", 1000, _fresh_curve())
    assert fill.paper is True
    assert fill.side == "BUY"
    assert fill.mint == "MINT_X"
    assert fill.fill_units == 1000
