"""Phase 2 feature engineering unit tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from polymarket_analytics.features import (
    assign_price_bucket,
    compute_time_to_resolution,
    compute_trade_features,
    run_compute_features,
)
from polymarket_analytics.ingest import run_ingest
from polymarket_analytics.schema import PRICE_BUCKET_LABELS
from polymarket_analytics.store import connect

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


def _ts(hours: float) -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=hours)


def _synthetic_trades() -> pl.DataFrame:
    """Controlled tape: one token, linear price + known notionals/sizes."""
    rows = []
    # t=0..5h: price rises 0.40 → 0.50 (0.02/h), size=10 each hour
    for h in range(6):
        price = 0.40 + 0.02 * h
        size = 10.0
        rows.append(
            {
                "trade_id": f"t{h}",
                "condition_id": "c1",
                "token_id": "tokA",
                "traded_at": _ts(h),
                "price": price,
                "size": size,
                "notional": price * size,
            }
        )
    # Dense large-size burst in last hour (whale prints)
    for i, frac in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        rows.append(
            {
                "trade_id": f"b{i}",
                "condition_id": "c1",
                "token_id": "tokA",
                "traded_at": _ts(5 + frac),
                "price": 0.49,  # slight down vs prior 0.50 for divergence
                "size": 100.0,
                "notional": 49.0,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("traded_at").cast(pl.Datetime("us", "UTC"))
    )


def _synthetic_markets() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "condition_id": ["c1"],
            "resolved_at": [datetime(2024, 1, 3, tzinfo=timezone.utc)],
            "closed_at": [None],
            "end_date": [None],
        }
    ).with_columns(
        pl.col("resolved_at").cast(pl.Datetime("us", "UTC")),
        pl.col("closed_at").cast(pl.Datetime("us", "UTC")),
        pl.col("end_date").cast(pl.Datetime("us", "UTC")),
    )


def test_price_bucket_edges() -> None:
    prices = pl.DataFrame({"price": [0.02, 0.42, 0.87, 0.92, 0.97, 1.0]})
    out = prices.with_columns(assign_price_bucket(pl.col("price")))
    assert out["price_bucket"].to_list() == [
        "0.00-0.05",
        "0.40-0.50",
        "0.85-0.90",
        "0.90-0.95",
        "0.95-1.00",
        "0.95-1.00",
    ]
    assert set(PRICE_BUCKET_LABELS) >= set(out["price_bucket"].to_list())


def test_momentum_velocity_and_volatility() -> None:
    trades = _synthetic_trades()
    markets = _synthetic_markets()
    feats = compute_trade_features(trades, markets)

    # At t=5h (trade t5): window includes t0..t5 → Δp=0.10 over 5h → 0.02 / hour
    row = feats.filter(pl.col("trade_id") == "t5")
    assert row.height == 1
    mom = row["momentum_6h"][0]
    assert mom is not None
    assert abs(mom - 0.02) < 1e-6

    # Single-point windows have null/0 volatility; multi-point should be > 0
    assert row["volatility_6h"][0] is not None
    assert row["volatility_6h"][0] > 0


def test_volume_spike_ratio() -> None:
    trades = _synthetic_trades()
    feats = compute_trade_features(trades, _synthetic_markets())
    # Last burst trade: 1h volume dominated by 5x100 + some earlier
    last = feats.sort("traded_at").tail(1)
    spike = last["volume_spike_1h_24h"][0]
    assert spike is not None
    assert spike > 1.0  # elevated vs 24h average hourly


def test_time_to_resolution_hours() -> None:
    trades = _synthetic_trades()
    markets = _synthetic_markets()
    ttr = compute_time_to_resolution(trades, markets)
    row = ttr.filter(pl.col("trade_id") == "t0")
    # close 2024-01-03 00:00 - 2024-01-01 00:00 = 48h
    assert abs(row["time_to_resolution_hours"][0] - 48.0) < 1e-6


def test_whale_ratio_and_divergence() -> None:
    feats = compute_trade_features(_synthetic_trades(), _synthetic_markets())
    last = feats.sort("traded_at").tail(1)
    whale = last["whale_ratio"][0]
    assert whale is not None
    assert whale > 3.0  # 1h mean size elevated vs 24h median (~10)

    # Burst prints below prior price → negative 1h delta + whale → divergence
    assert last["price_volume_divergence"][0] is True

    # Decay-adjusted velocity uses price_delta_1h / sqrt(TTR+0.1)
    dav = last["decay_adjusted_velocity"][0]
    assert dav is not None
    assert dav < 0  # price declined in the 1h window into the burst


def test_end_to_end_compute_features_and_view(tmp_path: Path) -> None:
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
    result = run_compute_features(out, warehouse_path=wh, bootstrap_warehouse=True)
    assert int(result["rows"]) >= 18

    con = connect(wh)
    try:
        n = con.execute("SELECT COUNT(*) FROM v_trade_features").fetchone()[0]
        assert n > 0
        cols = {
            r[0]
            for r in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'v_trade_features'"
            ).fetchall()
        }
        for required in (
            "momentum_1h",
            "volatility_24h",
            "volume_spike_1h_24h",
            "price_bucket",
            "time_to_resolution_hours",
            "whale_ratio",
            "decay_adjusted_velocity",
            "price_volume_divergence",
            "token_won",
        ):
            assert required in cols
    finally:
        con.close()
