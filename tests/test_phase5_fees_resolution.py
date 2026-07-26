"""Phase 5: dynamic taker fees, resolution settlement, TP/SL."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

from polymarket_analytics.live_feed import MarketResolved, parse_market_event
from polymarket_analytics.paper_trader import (
    PaperConfig,
    PaperPosition,
    PaperTrader,
    dynamic_taker_fee,
    relative_taker_fee_rate,
    resolve_fee_rate,
)


def test_dynamic_fee_higher_relative_at_mid_than_high() -> None:
    """Parabolic schedule: fee/notional larger at $0.50 than at $0.85."""
    fee_rate = 0.07  # crypto
    shares = 100.0
    fee_50 = dynamic_taker_fee(shares, 0.50, fee_rate=fee_rate)
    fee_85 = dynamic_taker_fee(shares, 0.85, fee_rate=fee_rate)
    notional_50 = shares * 0.50
    notional_85 = shares * 0.85
    rel_50 = fee_50 / notional_50
    rel_85 = fee_85 / notional_85

    assert abs(fee_50 - shares * fee_rate * 0.50 * 0.50) < 1e-12
    assert abs(fee_85 - shares * fee_rate * 0.85 * 0.15) < 1e-12
    assert rel_50 > rel_85
    assert abs(rel_50 - relative_taker_fee_rate(0.50, fee_rate=fee_rate)) < 1e-12
    assert abs(rel_85 - relative_taker_fee_rate(0.85, fee_rate=fee_rate)) < 1e-12
    # Peak absolute fee at mid for fixed C
    assert fee_50 > fee_85


def test_category_fee_rates() -> None:
    assert resolve_fee_rate("crypto") == 0.07
    assert resolve_fee_rate("sports") == 0.05
    assert resolve_fee_rate("geopolitics") == 0.0
    assert resolve_fee_rate("crypto", fee_rate=0.25) == 0.25


def test_settlement_pnl_win_and_loss(tmp_path: Path) -> None:
    trader = PaperTrader(
        config=PaperConfig(
            bankroll=1_000.0,
            use_dynamic_fees=True,
            fee_category="crypto",
            spread_slippage_bps=0.0,
            cooldown_sec=0.0,
        ),
        journal_path=tmp_path / "j.json",
    )
    # Manual long at 0.50 with known size
    shares = 100.0
    px = 0.50
    fees = dynamic_taker_fee(shares, px, fee_rate=0.07)
    notional = shares * px
    trader.cash -= notional + fees
    win_pos = PaperPosition(
        position_id="win1",
        token_id="tok_yes",
        condition_id="cond1",
        strategy_label="test",
        entry_ts=time.time(),
        entry_price_raw=px,
        entry_price_fill=px,
        shares=shares,
        notional=notional,
        fees=fees,
        mark_price=px,
        fee_rate=0.07,
    )
    lose_pos = PaperPosition(
        position_id="lose1",
        token_id="tok_no",
        condition_id="cond2",
        strategy_label="test",
        entry_ts=time.time(),
        entry_price_raw=px,
        entry_price_fill=px,
        shares=shares,
        notional=notional,
        fees=fees,
        mark_price=px,
        fee_rate=0.07,
    )
    trader.cash -= notional + fees
    trader.positions.extend([win_pos, lose_pos])

    cash_before = trader.cash
    closed_win = trader.resolve_position("win1", token_won=True)
    assert closed_win is not None
    assert closed_win.exit_price == 1.0
    expected_win_pnl = 1.0 * shares - notional - fees
    assert abs(closed_win.realized_pnl - expected_win_pnl) < 1e-9
    assert closed_win.win is True
    assert abs(trader.cash - (cash_before + shares)) < 1e-9

    cash_mid = trader.cash
    closed_lose = trader.resolve_position("lose1", token_won=False)
    assert closed_lose is not None
    assert closed_lose.exit_price == 0.0
    expected_lose_pnl = 0.0 - notional - fees
    assert abs(closed_lose.realized_pnl - expected_lose_pnl) < 1e-9
    assert closed_lose.win is False
    assert abs(trader.cash - cash_mid) < 1e-9  # no proceeds on loss
    assert abs(trader.realized_pnl - (expected_win_pnl + expected_lose_pnl)) < 1e-9


def test_on_market_resolved_settles_condition(tmp_path: Path) -> None:
    trader = PaperTrader(
        config=PaperConfig(bankroll=500.0, spread_slippage_bps=0.0),
        journal_path=tmp_path / "j.json",
    )
    shares, px = 50.0, 0.40
    fees = dynamic_taker_fee(shares, px, fee_rate=trader.config.effective_fee_rate())
    notional = shares * px
    trader.cash -= notional + fees
    trader.positions.append(
        PaperPosition(
            position_id="p1",
            token_id="yes_tok",
            condition_id="c_abc",
            strategy_label="t",
            entry_ts=1.0,
            entry_price_raw=px,
            entry_price_fill=px,
            shares=shares,
            notional=notional,
            fees=fees,
            mark_price=px,
        )
    )
    closed = trader.on_market_resolved(
        condition_id="c_abc", winning_asset_id="yes_tok", exit_ts=2.0
    )
    assert len(closed) == 1
    assert closed[0].win is True
    assert closed[0].exit_reason == "resolution"


def test_parse_market_resolved_event() -> None:
    items = parse_market_event(
        {
            "event_type": "market_resolved",
            "market": "condXYZ",
            "winning_asset_id": "tokWinner",
            "timestamp": 1700000000000,
        }
    )
    assert len(items) == 1
    assert isinstance(items[0], MarketResolved)
    assert items[0].condition_id == "condXYZ"
    assert items[0].winning_asset_id == "tokWinner"


def test_take_profit_closes_position(tmp_path: Path) -> None:
    from polymarket_analytics.live_feed import LiveFeatures

    trader = PaperTrader(
        config=PaperConfig(
            bankroll=1_000.0,
            take_profit_pct=0.10,
            spread_slippage_bps=0.0,
            use_dynamic_fees=True,
            fee_category="geopolitics",  # zero fee for cleaner math
            resolve_poll_sec=10_000,
        ),
        journal_path=tmp_path / "j.json",
    )
    shares, px = 100.0, 0.50
    trader.cash -= shares * px
    trader.positions.append(
        PaperPosition(
            position_id="tp1",
            token_id="tok",
            condition_id="c",
            strategy_label="t",
            entry_ts=1.0,
            entry_price_raw=px,
            entry_price_fill=px,
            shares=shares,
            notional=shares * px,
            fees=0.0,
            mark_price=px,
        )
    )
    feat = LiveFeatures(
        token_id="tok",
        condition_id="c",
        price=0.56,  # +12%
        size=1.0,
        ts=2.0,
        momentum_1h=0.01,
        whale_ratio=1.0,
        n_trades_1h=1,
    )
    trader.on_features(feat)
    assert not trader.open_positions
    closed = [p for p in trader.positions if p.status == "take_profit"]
    assert len(closed) == 1
    assert closed[0].exit_reason == "take_profit"
    assert closed[0].realized_pnl is not None
    assert closed[0].realized_pnl > 0


def test_resolution_poll_uses_gamma(tmp_path: Path) -> None:
    trader = PaperTrader(
        config=PaperConfig(
            bankroll=500.0,
            resolve_poll_sec=0.0,
            fee_category="geopolitics",
            spread_slippage_bps=0.0,
        ),
        journal_path=tmp_path / "j.json",
    )
    shares, px = 10.0, 0.50
    trader.cash -= shares * px
    trader.positions.append(
        PaperPosition(
            position_id="r1",
            token_id="win_tok",
            condition_id="cond_poll",
            strategy_label="t",
            entry_ts=1.0,
            entry_price_raw=px,
            entry_price_fill=px,
            shares=shares,
            notional=shares * px,
            fees=0.0,
            mark_price=px,
        )
    )
    fake = {"resolved": True, "winning_token_id": "win_tok", "closed": True, "raw": {}}
    with patch(
        "polymarket_analytics.paper_trader.fetch_market_resolution", return_value=fake
    ):
        closed = trader.maybe_poll_resolutions(now=time.time())
    assert len(closed) == 1
    assert closed[0].win is True
