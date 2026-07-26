"""Phase 4: live feed parsing + paper trader unit tests (no live orders)."""

from __future__ import annotations

import json
from pathlib import Path

from polymarket_analytics.live_feed import (
    LiveFeatureEngine,
    LiveTradeTick,
    TopOfBook,
    inject_demo_ticks,
    parse_market_event,
)
from polymarket_analytics.paper_trader import (
    PaperConfig,
    PaperTrader,
    kelly_fraction,
    load_oos_strategies,
    price_bucket_label,
)


def test_parse_last_trade_price_event() -> None:
    payload = {
        "event_type": "last_trade_price",
        "asset_id": "tok123",
        "market": "condABC",
        "price": "0.45",
        "size": "25",
        "side": "BUY",
        "timestamp": "1700000000000",
    }
    items = parse_market_event(payload)
    assert len(items) == 1
    tick = items[0]
    assert isinstance(tick, LiveTradeTick)
    assert tick.token_id == "tok123"
    assert tick.price == 0.45
    assert tick.size == 25.0


def test_parse_best_bid_ask_event() -> None:
    payload = {
        "event_type": "best_bid_ask",
        "asset_id": "tok123",
        "market": "condABC",
        "best_bid": "0.44",
        "best_ask": "0.46",
        "timestamp": 1700000000,
    }
    items = parse_market_event(payload)
    assert len(items) == 1
    book = items[0]
    assert isinstance(book, TopOfBook)
    assert book.best_bid == 0.44
    assert book.best_ask == 0.46
    assert abs(book.mid - 0.45) < 1e-9


def test_live_feature_engine_whale_and_momentum() -> None:
    engine = LiveFeatureEngine()
    feats = inject_demo_ticks(engine, n=30, base_price=0.45)
    assert len(feats) == 30
    last = feats[-1]
    assert last.whale_ratio is not None
    assert last.whale_ratio > 3.0
    assert last.momentum_1h is not None
    assert last.n_trades_1h >= 2


def test_price_bucket_mid() -> None:
    assert price_bucket_label(0.45) == "0.40-0.50"
    assert price_bucket_label(0.02) == "0.00-0.05"
    assert price_bucket_label(0.97) == "0.95-1.00"


def test_kelly_fraction_positive_edge() -> None:
    f = kelly_fraction(0.6, payout_odds=1.0, fraction=0.25, max_fraction=0.05)
    assert 0 < f <= 0.05


def test_kelly_fraction_no_edge() -> None:
    assert kelly_fraction(0.4, payout_odds=1.0) == 0.0


def test_paper_trader_fills_on_whale_mid_bucket(tmp_path: Path) -> None:
    journal = tmp_path / "paper_trades.json"
    cfg = PaperConfig(
        bankroll=10_000.0,
        min_oos_ev_pct=10.0,
        require_persists=False,
        cooldown_sec=0.0,
        spread_slippage_bps=50.0,
        kelly_fraction=0.25,
        max_position_pct=0.05,
    )
    trader = PaperTrader(
        config=cfg,
        strategies=load_oos_strategies(None, min_oos_ev_pct=10.0),
        journal_path=journal,
    )
    engine = LiveFeatureEngine()
    engine.on_book(
        TopOfBook(
            token_id="demo_token",
            condition_id="demo_condition",
            best_bid=0.41,
            best_ask=0.43,
            ts=0.0,
        )
    )
    feats = inject_demo_ticks(engine, n=40, base_price=0.35)
    for feat in feats:
        trader.on_features(feat)

    assert trader.ticks_seen == 40
    assert trader.signals_fired >= 1
    assert trader.fills >= 1
    assert journal.exists()
    payload = json.loads(journal.read_text())
    assert payload["fills"] >= 1
    assert payload["open_positions"] or payload["closed_positions"]
    pos = trader.open_positions[0]
    assert pos.entry_price_fill >= pos.entry_price_raw * 0.99


def test_paper_trader_resolve_updates_realized(tmp_path: Path) -> None:
    trader = PaperTrader(
        config=PaperConfig(cooldown_sec=0.0),
        strategies=load_oos_strategies(None),
        journal_path=tmp_path / "j.json",
    )
    engine = LiveFeatureEngine()
    for feat in inject_demo_ticks(engine, n=40, base_price=0.35):
        trader.on_features(feat)
    assert trader.open_positions
    pid = trader.open_positions[0].position_id
    closed = trader.resolve_position(pid, token_won=True)
    assert closed is not None
    assert closed.status == "resolved"
    assert closed.realized_pnl is not None
    assert closed.realized_pnl > 0
    assert trader.realized_pnl > 0


def test_load_oos_strategies_filters_min_ev(tmp_path: Path) -> None:
    report = {
        "comparisons": [
            {
                "label": "bucket=0.40-0.50 AND whale>3 AND side=BUY",
                "params": {
                    "price_bucket": "0.40-0.50",
                    "min_volume_spike": 2.0,
                    "min_whale_ratio": 3.0,
                    "require_price_volume_divergence": False,
                    "momentum_1h": "any",
                    "momentum_6h": "any",
                    "max_time_to_resolution_hours": 48.0,
                    "side": "BUY",
                },
                "oos_ev_pct": 15.5,
                "oos_win_rate": 0.5,
                "persists": True,
            },
            {
                "label": "noise",
                "params": {
                    "price_bucket": "0.90-0.95",
                    "min_whale_ratio": None,
                    "momentum_1h": "any",
                    "momentum_6h": "any",
                    "side": "BUY",
                },
                "oos_ev_pct": 1.0,
                "oos_win_rate": 0.9,
                "persists": True,
            },
        ]
    }
    path = tmp_path / "oos_edge_report.json"
    path.write_text(json.dumps(report))
    strats = load_oos_strategies(path, min_oos_ev_pct=10.0, require_persists=True)
    assert len(strats) == 1
    params, ev, wr = strats[0]
    assert params.price_bucket == "0.40-0.50"
    assert params.min_whale_ratio == 3.0
    assert params.min_volume_spike is None
    assert params.max_time_to_resolution_hours is None
    assert ev == 15.5
    assert wr == 0.5


def test_cli_paper_trade_demo(tmp_path: Path) -> None:
    from polymarket_analytics.cli import main

    report = {
        "comparisons": [
            {
                "label": "bucket=0.30-0.40 AND whale>3 AND mom1h=pos AND side=BUY",
                "params": {
                    "price_bucket": "0.30-0.40",
                    "min_volume_spike": None,
                    "min_whale_ratio": 3.0,
                    "require_price_volume_divergence": False,
                    "momentum_1h": "pos",
                    "momentum_6h": "any",
                    "max_time_to_resolution_hours": None,
                    "side": "BUY",
                },
                "oos_ev_pct": 15.0,
                "oos_win_rate": 0.55,
                "persists": True,
            }
        ]
    }
    oos_path = tmp_path / "oos_edge_report.json"
    oos_path.write_text(json.dumps(report))
    journal = tmp_path / "paper_trades.json"
    rc = main(
        [
            "paper-trade",
            "--demo",
            "--min-ev",
            "0.10",
            "--oos-report",
            str(oos_path),
            "--journal",
            str(journal),
            "--demo-ticks",
            "35",
            "--demo-sleep",
            "0",
        ]
    )
    assert rc == 0
    assert journal.exists()
    data = json.loads(journal.read_text())
    assert data["ticks_seen"] == 35
    assert data["fills"] >= 1
