"""Phase 3 edge finder + backtester unit tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from polymarket_analytics.backtest import (
    StrategyParams,
    apply_strategy_filter,
    compute_edge_stats,
    find_edges,
    run_backtest,
    run_find_edges,
    simulate_strategy,
)
from polymarket_analytics.features import run_compute_features
from polymarket_analytics.ingest import run_ingest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


def _feature_frame() -> pl.DataFrame:
    """Controlled resolved-trade feature rows for EV / PnL math."""
    # 5 wins @ 0.90, 5 losses @ 0.90 → empirical=0.5, implied=0.9, EV%=-40
    rows = []
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(10):
        rows.append(
            {
                "trade_id": f"a{i}",
                "condition_id": "cA",
                "token_id": "tokA",
                "side": "BUY",
                "price": 0.90,
                "token_won": i < 5,
                "price_bucket": "0.90-0.95",
                "volume_spike_1h_24h": 2.5 if i < 6 else 1.0,
                "momentum_1h": 0.01 if i % 2 == 0 else -0.01,
                "momentum_6h": 0.02,
                "time_to_resolution_hours": 12.0 if i < 8 else 100.0,
                "traded_at": t0.replace(hour=i % 24),
            }
        )
    # Positive-EV pocket: 4 wins @ 0.40 in bucket 0.40-0.50, spike>2, ttr<24
    for i in range(4):
        rows.append(
            {
                "trade_id": f"b{i}",
                "condition_id": "cB",
                "token_id": "tokB",
                "side": "BUY",
                "price": 0.40,
                "token_won": True,
                "price_bucket": "0.40-0.50",
                "volume_spike_1h_24h": 3.0,
                "momentum_1h": 0.05,
                "momentum_6h": 0.01,
                "time_to_resolution_hours": 6.0,
                "traded_at": t0.replace(day=2, hour=i),
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("traded_at").cast(pl.Datetime("us", "UTC"))
    )


def test_ev_math_empirical_minus_implied() -> None:
    df = _feature_frame().filter(pl.col("price_bucket") == "0.90-0.95")
    params = StrategyParams(price_bucket="0.90-0.95", side=None)
    sliced = apply_strategy_filter(df, params)
    stats = compute_edge_stats(sliced, params)
    assert stats.n == 10
    assert abs(stats.empirical_win_rate - 0.5) < 1e-9
    assert abs(stats.implied_win_rate - 0.9) < 1e-9
    assert abs(stats.ev_pct - (-40.0)) < 1e-6


def test_filter_volume_spike_and_ttr() -> None:
    df = _feature_frame()
    params = StrategyParams(
        price_bucket="0.90-0.95",
        min_volume_spike=2.0,
        max_time_to_resolution_hours=24.0,
        side=None,
    )
    sliced = apply_strategy_filter(df, params)
    # i<6 have spike 2.5; among those i<8 have ttr 12 → i=0..5
    assert sliced.height == 6
    assert (sliced["volume_spike_1h_24h"] > 2.0).all()
    assert (sliced["time_to_resolution_hours"] < 24.0).all()


def test_simulate_strategy_pnl_sharpe_drawdown() -> None:
    df = _feature_frame()
    params = StrategyParams(
        price_bucket="0.40-0.50",
        min_volume_spike=2.0,
        max_time_to_resolution_hours=48.0,
        momentum_1h="pos",
        side="BUY",
    )
    result = simulate_strategy(df, params)
    assert result.n == 4
    assert result.win_rate == 1.0
    # Each trade: pnl = 1 - 0.40 = 0.60
    assert abs(result.total_pnl - 2.4) < 1e-9
    assert abs(result.avg_pnl - 0.6) < 1e-9
    assert abs(result.ev_pct - 60.0) < 1e-6  # 100*(1.0-0.40)
    # Constant ROI → undefined Sharpe → 0 by convention
    assert result.sharpe == 0.0
    assert result.max_drawdown == 0.0  # monotonically rising equity
    assert len(result.equity_curve) == 4
    assert abs(result.equity_curve[-1] - 2.4) < 1e-9


def test_sharpe_positive_with_mixed_returns() -> None:
    df = _feature_frame().filter(pl.col("price_bucket") == "0.90-0.95")
    result = simulate_strategy(df, StrategyParams(price_bucket="0.90-0.95", side=None))
    # Mixed wins/losses → non-zero return variance and defined Sharpe
    assert result.n == 10
    assert result.sharpe != 0.0
    assert result.max_drawdown > 0.0


def test_find_edges_ranks_positive_ev() -> None:
    df = _feature_frame()
    edges = find_edges(
        df,
        min_samples=4,
        price_buckets=["0.40-0.50", "0.90-0.95"],
        spike_thresholds=(None, 2.0),
        whale_thresholds=(None,),
        ttr_bounds=(None, 24.0),
        momentum_1h_signs=("any", "pos"),
        momentum_6h_signs=("any",),
        side=None,
    )
    assert edges.height >= 1
    top = edges.row(0, named=True)
    assert top["ev_pct"] > 0
    assert top["price_bucket"] == "0.40-0.50"


def test_end_to_end_cli_path_on_fixtures(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "parquet"
    wh = tmp_path / "warehouse.duckdb"
    (raw / "trades").mkdir(parents=True)
    (raw / "markets").mkdir(parents=True)
    (raw / "trades" / "sample_trades.json").write_text(
        (FIXTURES / "sample_trades.json").read_text()
    )
    (raw / "markets" / "sample_markets.json").write_text(
        (FIXTURES / "sample_markets.json").read_text()
    )

    run_ingest(raw, out, bootstrap_warehouse=True, warehouse_path=wh)
    run_compute_features(out, warehouse_path=wh, bootstrap_warehouse=True)

    edges = run_find_edges(wh, out, min_samples=1, top_k=20)
    assert edges.height >= 1
    assert {"n", "empirical_win_rate", "implied_win_rate", "ev_pct"} <= set(edges.columns)

    bt = run_backtest(
        wh,
        StrategyParams(price_bucket=None, min_volume_spike=None, side="BUY"),
        out,
    )
    assert bt.n >= 1
    assert isinstance(bt.total_pnl, float)
    assert isinstance(bt.max_drawdown, float)
