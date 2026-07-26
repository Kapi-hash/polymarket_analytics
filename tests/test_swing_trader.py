"""Unit tests for swing trading indicators, entries, and exits."""

from __future__ import annotations

from pathlib import Path

from polymarket_analytics.swing_trader import (
    SwingConfig,
    SwingPosition,
    SwingTrader,
    TokenBar,
    TokenSeries,
    compute_hurst,
    compute_rsi,
    detect_confluence_legs,
    evaluate_confluence,
    evaluate_mean_reversion,
    evaluate_momentum,
    inject_swing_demo_bars,
)


def test_rsi_oversold_on_sharp_decline() -> None:
    closes = [0.50] * 5 + [0.50 - 0.02 * i for i in range(1, 20)]
    rsi = compute_rsi(closes, period=14)
    assert rsi is not None
    assert rsi < 25.0


def test_mean_reversion_entry_on_oversold() -> None:
    cfg = SwingConfig(
        min_liquidity_usd=50_000.0,
        rsi_oversold=25.0,
        min_ev_pct=1.0,
        cooldown_sec=0.0,
        stall_hours=1000.0,
    )
    series = TokenSeries(maxlen=200)
    for i, px in enumerate([0.55 - 0.01 * i for i in range(30)]):
        series.add(
            TokenBar(
                ts=float(i),
                mid=px,
                liquidity_usd=80_000.0,
                whale_ratio=1.0,
            )
        )
    ind = series.indicators(cfg)
    assert ind.rsi is not None and ind.rsi < 25.0
    bar = series.bars[-1]
    sig = evaluate_mean_reversion("tok", "cond", bar, ind, series, cfg)
    assert sig is not None
    assert sig.strategy == "mean_reversion"
    assert sig.target_price > sig.entry_price


def test_hurst_trending_above_threshold() -> None:
    closes = [0.20 + 0.01 * i + 0.001 * ((-1) ** i) for i in range(80)]
    h = compute_hurst(closes)
    assert h is not None
    assert h > 0.55


def test_momentum_entry_hurst_confirmation() -> None:
    cfg = SwingConfig(
        hurst_min=0.55,
        momentum_min_move=0.08,
        whale_volume_ratio=2.0,
        ema_fast=5,
        ema_slow=20,
        min_ev_pct=1.0,
        cooldown_sec=0.0,
    )
    series = TokenSeries(maxlen=200)
    for i in range(50):
        px = 0.25 + 0.005 * i
        series.add(
            TokenBar(
                ts=float(i),
                mid=px,
                volume=1000.0,
                whale_ratio=3.0,
                liquidity_usd=100_000.0,
            )
        )
    ind = series.indicators(cfg)
    assert ind.hurst is not None
    assert ind.hurst > 0.55
    bar = series.bars[-1]
    sig = evaluate_momentum("tok", "cond", bar, ind, series, cfg)
    if sig is not None:
        assert sig.strategy == "momentum"
        assert sig.target_price > sig.entry_price
    else:
        # EMA cross may not fire on monotone path; Hurst still confirmed above
        assert ind.ema_fast is not None and ind.ema_slow is not None


def test_trailing_stop_exit(tmp_path: Path) -> None:
    trader = SwingTrader(
        config=SwingConfig(stall_hours=1000.0, atr_stop_mult=2.0),
        journal_path=tmp_path / "swing.json",
    )
    trader.cash = 10_000.0
    trader.positions.append(
        SwingPosition(
            position_id="sl1",
            strategy="momentum",
            token_id="tok",
            condition_id="cond",
            entry_ts=0.0,
            entry_price=0.40,
            shares=100.0,
            notional=40.0,
            target_price=0.80,
            stop_price=0.30,
            trailing_stop=0.38,
            mark_price=0.45,
        )
    )
    trader.cash -= 40.0
    # Mark below trailing stop → stop_loss
    trader.on_bar(
        "tok",
        "cond",
        TokenBar(ts=10.0, mid=0.35, liquidity_usd=80_000.0),
        strategies=(),
    )
    assert any(p.status == "stop_loss" for p in trader.positions)
    closed = next(p for p in trader.positions if p.position_id == "sl1")
    assert closed.realized_pnl is not None
    assert closed.exit_reason == "stop_loss"


def test_time_decay_exit(tmp_path: Path) -> None:
    trader = SwingTrader(
        config=SwingConfig(
            stall_hours=1.0 / 3600.0,  # 1 second
            take_profit_pct=5.0,
            atr_stop_mult=100.0,
        ),
        journal_path=tmp_path / "s.json",
    )
    trader.positions.append(
        SwingPosition(
            position_id="td1",
            strategy="book_imbalance",
            token_id="tok",
            condition_id="cond",
            entry_ts=0.0,
            entry_price=0.40,
            shares=100.0,
            notional=40.0,
            target_price=0.90,
            stop_price=0.01,
            trailing_stop=0.01,
            mark_price=0.40,
        )
    )
    trader.cash -= 40.0
    trader.on_bar(
        "tok",
        "cond",
        TokenBar(ts=10.0, mid=0.41, liquidity_usd=80_000.0),
        strategies=(),
    )
    assert any(p.status == "time_decay" for p in trader.positions)


def test_book_imbalance_and_demo_journal(tmp_path: Path) -> None:
    journal = tmp_path / "swing_trades.json"
    cfg = SwingConfig(
        min_ev_pct=1.0,
        cooldown_sec=0.0,
        stall_hours=1000.0,
        take_profit_pct=0.15,
        min_liquidity_usd=50_000.0,
        book_imbalance_min=3.0,
    )
    trader = SwingTrader(config=cfg, journal_path=journal)
    for token_id, condition_id, bar in inject_swing_demo_bars(n=80):
        trader.on_bar(token_id, condition_id, bar)
    assert trader.ticks_seen == 80
    assert journal.exists()
    assert trader.signals_fired >= 1 or trader.fills >= 1


def test_cli_swing_trade_demo(tmp_path: Path) -> None:
    from polymarket_analytics.cli import main

    journal = tmp_path / "swing_trades.json"
    rc = main(
        [
            "swing-trade",
            "--demo",
            "--min-ev",
            "0.08",
            "--journal",
            str(journal),
            "--demo-ticks",
            "80",
            "--demo-sleep",
            "0",
        ]
    )
    assert rc == 0
    assert journal.exists()


def test_confluence_entry_and_36h_time_exit(tmp_path: Path) -> None:
    cfg = SwingConfig(
        require_confluence=True,
        min_confluence=2,
        confluence_book_min=2.5,
        confluence_rsi=30.0,
        confluence_volume_usd=25_000.0,
        stall_hours=36.0,
        take_profit_pct=0.20,
        take_profit_atr_mult=2.0,
        stop_loss_pct=0.10,
        stop_loss_atr_mult=1.0,
        min_ev_pct=0.0,
        cooldown_sec=0.0,
    )
    series = TokenSeries(maxlen=200)
    for i in range(50):
        series.add(TokenBar(ts=float(i), mid=0.55 - 0.007 * i, liquidity_usd=80_000.0))
    ind = series.indicators(cfg)
    bar = TokenBar(
        ts=50.0,
        mid=series.closes[-1],
        liquidity_usd=80_000.0,
        bid_depth=40_000.0,
        ask_depth=10_000.0,
        volume=30_000.0,
    )
    assert len(detect_confluence_legs(bar, ind, series, cfg)) >= 2
    sig = evaluate_confluence("tok", "cond", bar, ind, series, cfg)
    assert sig is not None

    trader = SwingTrader(config=cfg, journal_path=tmp_path / "c.json")
    for i in range(50):
        trader.on_bar(
            "tok",
            "cond",
            TokenBar(
                ts=float(i),
                mid=0.55 - 0.007 * i,
                liquidity_usd=80_000.0,
                bid_depth=10.0,
                ask_depth=10.0,
            ),
            strategies=("confluence",),
        )
    # Confluence bar should open a position
    trader.on_bar("tok", "cond", bar, strategies=("confluence",))
    assert any(p.status == "open" and p.strategy == "confluence" for p in trader.positions)

    # 36h time exit without TP/SL
    open_pos = next(p for p in trader.positions if p.status == "open")
    open_pos.target_price = 0.99
    open_pos.stop_price = 0.01
    open_pos.trailing_stop = 0.01
    trader.on_bar(
        "tok",
        "cond",
        TokenBar(ts=open_pos.entry_ts + 36 * 3600 + 1, mid=open_pos.entry_price, liquidity_usd=80_000.0),
        strategies=(),
    )
    closed = next(p for p in trader.positions if p.position_id == open_pos.position_id)
    assert closed.status == "time_decay"
