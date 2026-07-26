"""Multi-profile incubator + confluence swing tests."""

from __future__ import annotations

from pathlib import Path

from polymarket_analytics.live_feed import LiveFeatureEngine, inject_demo_ticks
from polymarket_analytics.profiles import (
    MultiProfileIncubator,
    format_profile_leaderboard,
    load_profile_journals,
)
from polymarket_analytics.swing_trader import (
    SwingConfig,
    SwingTrader,
    TokenBar,
    TokenSeries,
    detect_confluence_legs,
    evaluate_confluence,
)


def test_profile_bankroll_isolation(tmp_path: Path) -> None:
    inc = MultiProfileIncubator.create(bankroll=10_000.0, min_ev_pct=10.0, data_dir=tmp_path)
    assert len(inc.profiles) == 4
    equities = [rt.equity() for rt in inc.profiles]
    assert all(abs(e - 10_000.0) < 1e-6 for e in equities)
    # Debit one profile only
    whale = next(r for r in inc.profiles if r.spec.name == "profile_whale_midbucket")
    assert whale.paper is not None
    whale.paper.cash -= 500.0
    assert abs(whale.equity() - 9_500.0) < 1e-6
    others = [r.equity() for r in inc.profiles if r is not whale]
    assert all(abs(e - 10_000.0) < 1e-6 for e in others)


def test_journals_written_separately(tmp_path: Path) -> None:
    inc = MultiProfileIncubator.create(bankroll=10_000.0, min_ev_pct=5.0, data_dir=tmp_path)
    engine = LiveFeatureEngine()
    for feat in inject_demo_ticks(engine, n=40, base_price=0.45):
        feat = type(feat)(**{**feat.__dict__, "best_bid": feat.price - 0.01, "best_ask": feat.price + 0.005})
        inc.on_features(feat)
    for name in (
        "paper_trades_whale.json",
        "paper_trades_rsi.json",
        "paper_trades_momentum.json",
        "paper_trades_swing.json",
    ):
        assert (tmp_path / name).exists()
    rows = inc.comparison_rows()
    assert len(rows) == 4
    board = format_profile_leaderboard(rows)
    assert "MULTI-PROFILE" in board


def test_confluence_requires_two_legs() -> None:
    cfg = SwingConfig(
        require_confluence=True,
        min_confluence=2,
        confluence_book_min=2.5,
        confluence_rsi=30.0,
        confluence_volume_usd=25_000.0,
        min_ev_pct=1.0,
        stall_hours=36.0,
    )
    series = TokenSeries(maxlen=200)
    # Sharp decline → oversold RSI (stay declining; no bounce that lifts RSI)
    for i in range(50):
        series.add(TokenBar(ts=float(i), mid=0.55 - 0.007 * i, liquidity_usd=80_000.0))
    ind = series.indicators(cfg)
    # Balanced book → only RSI leg → no entry
    bar = TokenBar(
        ts=50.0,
        mid=series.closes[-1],
        liquidity_usd=80_000.0,
        bid_depth=10.0,
        ask_depth=10.0,
        volume=30_000.0,
    )
    legs = detect_confluence_legs(bar, ind, series, cfg)
    assert len(legs) < 2
    assert evaluate_confluence("t", "c", bar, ind, series, cfg) is None
    # Strong buy-side depth + RSI → 2+ legs → entry
    bar2 = TokenBar(
        ts=51.0,
        mid=series.closes[-1],
        liquidity_usd=80_000.0,
        bid_depth=50_000.0,
        ask_depth=10_000.0,
        volume=30_000.0,
    )
    legs2 = detect_confluence_legs(bar2, ind, series, cfg)
    assert len(legs2) >= 2
    assert "signal_liquidity" in {n for n, _ in legs2}
    assert "signal_rsi" in {n for n, _ in legs2}
    sig = evaluate_confluence("t", "c", bar2, ind, series, cfg)
    assert sig is not None
    assert sig.strategy == "confluence"
    assert "confluence" in sig.reason
    assert sig.target_price > sig.entry_price
    assert sig.stop_price < sig.entry_price


def test_confluence_tp_and_sl(tmp_path: Path) -> None:
    trader = SwingTrader(
        config=SwingConfig(
            require_confluence=True,
            min_confluence=2,
            stall_hours=36.0,
            take_profit_pct=0.15,
            take_profit_atr_mult=2.0,
            stop_loss_pct=0.10,
            stop_loss_atr_mult=1.0,
            min_ev_pct=0.0,
            cooldown_sec=0.0,
        ),
        journal_path=tmp_path / "swing.json",
    )
    from polymarket_analytics.swing_trader import SwingPosition

    trader.positions.append(
        SwingPosition(
            position_id="c1",
            strategy="confluence",
            token_id="tok",
            condition_id="cond",
            entry_ts=0.0,
            entry_price=0.40,
            shares=100.0,
            notional=40.0,
            target_price=0.48,
            stop_price=0.36,
            trailing_stop=0.36,
            mark_price=0.40,
        )
    )
    trader.cash -= 40.0
    trader.on_bar("tok", "cond", TokenBar(ts=1.0, mid=0.49, liquidity_usd=80_000.0), strategies=())
    assert any(p.status == "take_profit" for p in trader.positions)

    trader.positions.append(
        SwingPosition(
            position_id="c2",
            strategy="confluence",
            token_id="tok2",
            condition_id="cond",
            entry_ts=2.0,
            entry_price=0.40,
            shares=100.0,
            notional=40.0,
            target_price=0.80,
            stop_price=0.36,
            trailing_stop=0.36,
            mark_price=0.40,
        )
    )
    trader.cash -= 40.0
    trader.on_bar("tok2", "cond", TokenBar(ts=3.0, mid=0.35, liquidity_usd=80_000.0), strategies=())
    assert any(p.position_id == "c2" and p.status == "stop_loss" for p in trader.positions)


def test_cli_compare(tmp_path: Path) -> None:
    from polymarket_analytics.cli import main

    MultiProfileIncubator.create(bankroll=10_000.0, min_ev_pct=10.0, data_dir=tmp_path)
    rc = main(["paper-trade", "--compare", "--journal", str(tmp_path / "paper_trades_whale.json")])
    assert rc == 0
