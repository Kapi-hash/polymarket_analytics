"""Regression coverage for overnight pipeline integrity repairs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from polymarket_analytics.collectors.book_collector import (
    NORMALIZED_SCHEMA,
    parse_book_message,
    rebuild_normalized_from_raw,
    write_normalized_rows,
)
from polymarket_analytics.research.historical_fees import lookup_fee_regime
from polymarket_analytics.research.overnight_backfill import (
    _market_matches_year,
    _normalize_trade,
    _trade_id,
    year_bounds_utc,
)
from polymarket_analytics.research.overnight_gate import evaluate_outcome_gate
from polymarket_analytics.research.outcome_sweep import run_outcome_sweep
from polymarket_analytics.features import compute_time_to_resolution


def test_market_year_filter_accepts_closed_or_end_date() -> None:
    assert _market_matches_year({"closedTime": "2024-01-03T01:02:03Z"}, 2024)
    assert _market_matches_year({"endDate": "2024-12-31"}, 2024)
    assert not _market_matches_year({"closedTime": "2023-12-31T23:59:59Z"}, 2024)


def test_trade_id_includes_asset() -> None:
    assert _trade_id({"transactionHash": "0xabc", "logIndex": 7, "asset": "99"}, 2) == "0xabc_7_99"


def test_normalize_trade_rejects_off_year_and_wrong_condition() -> None:
    good = {
        "timestamp": int(datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp()),
        "conditionId": "0xaaa",
        "asset": "1",
        "price": 0.4,
        "size": 2,
        "side": "BUY",
        "transactionHash": "0x1",
    }
    assert _normalize_trade(good, "0xaaa", "f.json", 2024, 0) is not None
    assert _normalize_trade(good, "0xbbb", "f.json", 2024, 0) is None
    bad_year = dict(good)
    bad_year["timestamp"] = int(datetime(2026, 7, 27, tzinfo=timezone.utc).timestamp())
    assert _normalize_trade(bad_year, "0xaaa", "f.json", 2024, 0) is None


def test_year_bounds_utc() -> None:
    start, end = year_bounds_utc(2025)
    assert start.year == 2025 and end.year == 2025
    assert start.month == 1 and end.month == 12


def test_gate_rejects_condition_fallback_and_missing_event_id() -> None:
    frame = pl.DataFrame(
        {
            "trade_id": [f"t{i}" for i in range(10)],
            "condition_id": [f"c{i}" for i in range(10)],
            "token_won": [True] * 10,
            "fee_confidence": ["exact"] * 10,
            "traded_at": [datetime(2023, 1, 1, tzinfo=timezone.utc)] * 10,
        }
    )
    result = evaluate_outcome_gate(frame, train_end_exclusive=None, require_expansion=False)
    assert result["decision"] == "BLOCKED"
    assert any("event_id" in r for r in result["reasons"])


def test_gate_rejects_zero_locked_test() -> None:
    n = 120
    frame = pl.DataFrame(
        {
            "trade_id": [f"t{i}" for i in range(n)],
            "event_id": [f"e{i % 60}" for i in range(n)],
            "condition_id": [f"c{i}" for i in range(n)],
            "token_won": [True] * n,
            "fee_confidence": ["exact"] * n,
            "traded_at": [datetime(2023, 1, 1, tzinfo=timezone.utc)] * n,
        }
    )
    result = evaluate_outcome_gate(
        frame,
        min_events=50,
        min_train_events=10,
        min_test_events=20,
        train_end_exclusive="2025-01-01T00:00:00+00:00",
        baseline_feature_rows=10,
        require_expansion=True,
    )
    assert result["decision"] == "BLOCKED"
    assert any("locked-test" in r for r in result["reasons"])


def test_gate_rejects_baseline_fallback_flag() -> None:
    frame = pl.DataFrame(
        {
            "trade_id": ["a"],
            "event_id": ["e1"],
            "token_won": [True],
            "fee_confidence": ["exact"],
            "traded_at": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
        }
    )
    result = evaluate_outcome_gate(frame, used_baseline_fallback=True, train_end_exclusive=None)
    assert result["decision"] == "BLOCKED"


def test_feature_ttr_without_literal_closed_at() -> None:
    trades = pl.DataFrame(
        {
            "trade_id": ["t1"],
            "condition_id": ["c1"],
            "traded_at": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
        }
    )
    markets = pl.DataFrame(
        {
            "condition_id": ["c1"],
            "resolved_at": [datetime(2024, 2, 1, tzinfo=timezone.utc)],
            "end_date": [datetime(2024, 2, 1, tzinfo=timezone.utc)],
        }
    )
    out = compute_time_to_resolution(trades, markets)
    assert out.height == 1
    assert out["time_to_resolution_hours"][0] is not None


def test_fee_boundary_and_unknown_category_fallback() -> None:
    zero = lookup_fee_regime("2026-01-05", category="crypto")
    assert zero["fee_regime"] == "zero_fee_historical"
    assert zero["fee_confidence"] == "exact"
    post = lookup_fee_regime("2026-01-06", category="crypto")
    assert post["fee_regime"] == "clob_v2_category"
    assert post["fee_confidence"] == "strongly_evidenced"
    unknown = lookup_fee_regime("2026-01-06", category=None)
    assert unknown["fee_confidence"] == "fallback"


def test_l2_schema_preserves_large_asset_ids(tmp_path: Path) -> None:
    huge = "98022490269692409998126496127597032490334070080325855126491859374983463996227"
    rows = [
        {
            "event_type": "book",
            "asset_id": huge,
            "best_bid": 0.4,
            "best_ask": 0.6,
            "price": None,
            "size": None,
            "side": None,
            "timestamp": "1",
            "received_at": "2026-01-01T00:00:00+00:00",
            "record_type": "book",
            "raw_keys": ["asset_id"],
            "session_id": "abc",
            "utc_date": "2026-01-01",
            "parse_ok": True,
        },
        {
            "event_type": "last_trade_price",
            "asset_id": huge,
            "best_bid": None,
            "best_ask": None,
            "price": 0.55,
            "size": 1.0,
            "side": "BUY",
            "timestamp": "2",
            "received_at": "2026-01-01T00:00:01+00:00",
            "record_type": "trade",
            "raw_keys": ["price"],
            "session_id": "abc",
            "utc_date": "2026-01-01",
            "parse_ok": True,
        },
    ]
    path = tmp_path / "norm.parquet"
    write_normalized_rows(rows, path)
    df = pl.read_parquet(path)
    assert df.schema["asset_id"] == pl.Utf8
    assert df["asset_id"][0] == huge


def test_rebuild_normalized_from_raw(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    huge = "123456789012345678901234567890123456789012345678901234567890"
    raw.write_text(
        json.dumps({"event_type": "book", "asset_id": huge, "best_bid": "0.1", "best_ask": "0.2", "timestamp": "9"})
        + "\n"
    )
    out = tmp_path / "out.parquet"
    report = rebuild_normalized_from_raw(raw, out, session_id="sess1")
    assert report["normalized_rows"] >= 1
    assert pl.read_parquet(out)["asset_id"].dtype == pl.Utf8


def test_parse_book_message_mixed_shapes() -> None:
    assert parse_book_message({"event_type": "price_change", "asset_id": "1", "best_bid": 0.2}) is not None
    assert parse_book_message({}) is None


def test_sweep_blocks_empty_locked_test() -> None:
    n = 80
    features = pl.DataFrame(
        {
            "trade_id": [f"t{i}" for i in range(n)],
            "condition_id": [f"c{i%10}" for i in range(n)],
            "event_id": [f"e{i%40}" for i in range(n)],
            "token_id": [f"tok{i%5}" for i in range(n)],
            "traded_at": [datetime(2023, 1, 1, tzinfo=timezone.utc)] * n,
            "price": [0.45] * n,
            "size": [1.0] * n,
            "token_won": [True] * n,
            "price_bucket": ["0.40-0.50"] * n,
            "volume_spike": [1.0] * n,
            "whale_ratio": [1.0] * n,
            "price_volume_divergence": [False] * n,
            "momentum_1h": [0.0] * n,
            "momentum_6h": [0.0] * n,
            "time_to_resolution_hours": [10.0] * n,
            "side": ["BUY"] * n,
        }
    )
    result = run_outcome_sweep(features, train_end_exclusive="2025-01-01T00:00:00+00:00")
    assert result["status"] == "BLOCKED"
    assert result["n_test_rows_purged"] == 0


def test_workflow_fail_closed_critical_paths() -> None:
    text = Path(".github/workflows/overnight-polymarket-research.yml").read_text()
    assert "|| true" not in text
    assert "set -euo pipefail" in text or "set -o pipefail" in text
    assert "research_authorized" in text
    # Critical research steps must not be continue-on-error; only optional final artifact download may.
    critical = []
    for block in text.split("- name:"):
        if any(k in block for k in ("Backfill year", "Collect L2", "Extract and merge", "Evaluate gate", "Run purged outcome")):
            if "continue-on-error: true" in block:
                critical.append(block.split("\n", 1)[0])
    assert critical == [], f"continue-on-error on critical steps: {critical}"
    assert "trade_features_canonical_expanded.parquet" in text
    assert "find data -name" not in text  # no silent baseline feature search
