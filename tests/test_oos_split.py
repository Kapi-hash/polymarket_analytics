"""Out-of-sample / temporal split unit tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from polymarket_analytics.backtest import (
    filter_by_traded_at,
    find_edges,
    parse_date_bound,
    run_oos_edge_validation,
)


def _ts(y: int, m: int, d: int, h: int = 12) -> datetime:
    return datetime(y, m, d, h, tzinfo=timezone.utc)


def _dated_feature_frame() -> pl.DataFrame:
    """Trades spanning Apr–Jun 2023 with a clear positive-EV pocket in both halves."""
    rows: list[dict] = []
    # In-sample (before May 1): 0.40-0.50 buys that mostly win
    for i in range(12):
        rows.append(
            {
                "trade_id": f"is_{i}",
                "condition_id": "c1",
                "token_id": "t1",
                "side": "BUY",
                "price": 0.42,
                "token_won": i < 9,  # 75% win
                "price_bucket": "0.40-0.50",
                "volume_spike_1h_24h": 2.5,
                "whale_ratio": 3.5,
                "momentum_1h": -0.01,
                "momentum_6h": 0.0,
                "time_to_resolution_hours": 24.0,
                "traded_at": _ts(2023, 4, 1 + (i % 28)),
            }
        )
    # Out-of-sample (on/after May 1): same setup, slightly weaker but still +EV
    for i in range(12):
        rows.append(
            {
                "trade_id": f"oos_{i}",
                "condition_id": "c1",
                "token_id": "t1",
                "side": "BUY",
                "price": 0.42,
                "token_won": i < 8,  # ~66.7% win
                "price_bucket": "0.40-0.50",
                "volume_spike_1h_24h": 2.5,
                "whale_ratio": 3.5,
                "momentum_1h": -0.01,
                "momentum_6h": 0.0,
                "time_to_resolution_hours": 24.0,
                "traded_at": _ts(2023, 5, 1 + (i % 28)),
            }
        )
    # Noise bucket that only wins in-sample (should decay OOS)
    for i in range(10):
        rows.append(
            {
                "trade_id": f"noise_is_{i}",
                "condition_id": "c2",
                "token_id": "t2",
                "side": "BUY",
                "price": 0.92,
                "token_won": True,
                "price_bucket": "0.90-0.95",
                "volume_spike_1h_24h": 1.0,
                "whale_ratio": 1.0,
                "momentum_1h": 0.02,
                "momentum_6h": 0.0,
                "time_to_resolution_hours": 24.0,
                "traded_at": _ts(2023, 4, 10 + i),
            }
        )
    for i in range(10):
        rows.append(
            {
                "trade_id": f"noise_oos_{i}",
                "condition_id": "c2",
                "token_id": "t2",
                "side": "BUY",
                "price": 0.92,
                "token_won": False,
                "price_bucket": "0.90-0.95",
                "volume_spike_1h_24h": 1.0,
                "whale_ratio": 1.0,
                "momentum_1h": 0.02,
                "momentum_6h": 0.0,
                "time_to_resolution_hours": 24.0,
                "traded_at": _ts(2023, 5, 10 + i),
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("traded_at").cast(pl.Datetime("us", "UTC"))
    )


def test_parse_date_bound_utc_midnight() -> None:
    dt = parse_date_bound("2023-05-01")
    assert dt is not None
    assert dt == datetime(2023, 5, 1, tzinfo=timezone.utc)


def test_filter_by_traded_at_half_open_isolation() -> None:
    df = _dated_feature_frame()
    split = "2023-05-01"
    train = filter_by_traded_at(df, end=split)
    test = filter_by_traded_at(df, start=split)

    assert train.height + test.height == df.height
    assert train.filter(pl.col("traded_at") >= parse_date_bound(split)).is_empty()
    assert test.filter(pl.col("traded_at") < parse_date_bound(split)).is_empty()
    # Boundary row on split date lands in test only
    assert "oos_0" in test["trade_id"].to_list()
    assert "oos_0" not in train["trade_id"].to_list()


def test_filter_start_end_window() -> None:
    df = _dated_feature_frame()
    window = filter_by_traded_at(df, start="2023-04-01", end="2023-05-01")
    assert window.height == train_count_apr(df)
    assert (window["traded_at"] < parse_date_bound("2023-05-01")).all()
    assert (window["traded_at"] >= parse_date_bound("2023-04-01")).all()


def train_count_apr(df: pl.DataFrame) -> int:
    return filter_by_traded_at(df, start="2023-04-01", end="2023-05-01").height


def test_oos_validation_compares_is_vs_oos(tmp_path: Path) -> None:
    df = _dated_feature_frame()
    report = run_oos_edge_validation(
        df,
        split_date="2023-05-01",
        min_samples=5,
        price_buckets=["0.40-0.50", "0.90-0.95"],
        spike_thresholds=(None, 2.0),
        whale_thresholds=(None,),
        ttr_bounds=(None,),
        momentum_1h_signs=("any",),
        top_k=10,
    )
    assert report["train_rows"] > 0
    assert report["test_rows"] > 0
    assert report["n_is_edges"] >= 1
    assert report["comparisons"]

    # Positive-EV mid bucket should appear and keep +EV OOS
    mid = [
        c
        for c in report["comparisons"]
        if c["params"]["price_bucket"] == "0.40-0.50" and c["is_n"] >= 5
    ]
    assert mid
    assert mid[0]["is_ev_pct"] > 0
    assert mid[0]["oos_n"] >= 5
    assert mid[0]["oos_ev_pct"] > 0

    # Write path works
    out = tmp_path / "oos_edge_report.json"
    import json

    out.write_text(json.dumps(report, indent=2, default=str))
    assert out.exists()


def test_find_edges_respects_prefiltered_train_only() -> None:
    df = _dated_feature_frame()
    train = filter_by_traded_at(df, end="2023-05-01")
    edges = find_edges(
        train,
        min_samples=5,
        price_buckets=["0.40-0.50"],
        spike_thresholds=(None,),
        whale_thresholds=(None,),
        ttr_bounds=(None,),
        momentum_1h_signs=("any",),
        side="BUY",
    )
    assert edges.height >= 1
    # All underlying train trades are before split — stats N cannot exceed train size
    assert int(edges["n"].max()) <= train.height
