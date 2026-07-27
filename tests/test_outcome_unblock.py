"""Tests for outcome-strategy research unblock modules."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from polymarket_analytics.collectors.book_collector import (
    detect_sequence_gap,
    parse_book_message,
    parse_trade_message,
)
from polymarket_analytics.research.duplicates import audit_duplicate_trades, canonicalize_trades
from polymarket_analytics.research.historical_fees import (
    FEE_MODEL_VERSION,
    compute_historical_fill_fee,
    lookup_fee_regime,
)
from polymarket_analytics.research.outcome_execution import (
    OutcomeExecConfig,
    adverse_fill_price,
    simulate_outcome_fill,
)
from polymarket_analytics.research.stats_valid import (
    bh_fdr,
    cluster_bootstrap_ci,
    deflated_sharpe_valid,
)


def test_historical_fee_zero_for_2023():
    regime = lookup_fee_regime("2023-04-15", category="crypto")
    assert regime["fee_regime"] == "zero_fee_historical"
    assert regime["fee_confidence"] == "exact"
    assert regime["taker_rate"] == 0.0

    fee = compute_historical_fill_fee(100.0, 0.5, "2023-06-01", role="taker", category="crypto")
    assert fee["fee"] == 0.0
    assert fee["fee_confidence"] == "exact"
    assert fee["fee_model_version"] == FEE_MODEL_VERSION


def test_historical_fee_current_regime_post_2026():
    regime = lookup_fee_regime("2026-02-01", category="crypto")
    assert regime["fee_regime"] == "clob_v2_category"
    assert regime["fee_confidence"] == "strongly_evidenced"
    assert regime["taker_rate"] == pytest.approx(0.07)


def test_dedupe_reduces_exact_duplicates():
    base = {
        "tx_hash": "0xabc",
        "token_id": "tok1",
        "price": 0.55,
        "size": 10.0,
        "traded_at": datetime(2023, 5, 1, 12, 0, tzinfo=timezone.utc),
        "side": "BUY",
    }
    df = pl.DataFrame(
        [
            {"trade_id": "t1", "ingest_at": datetime(2023, 5, 2, tzinfo=timezone.utc), **base},
            {"trade_id": "t2", "ingest_at": datetime(2023, 5, 3, tzinfo=timezone.utc), **base},
        ]
    )
    audit = audit_duplicate_trades(df)
    assert audit["n_duplicate_trade_ids"] == 0  # different trade_ids
    assert audit["n_rows"] == 2

    dup_df = pl.DataFrame(
        [
            {"trade_id": "dup", "ingest_at": datetime(2023, 5, 2, tzinfo=timezone.utc), **base},
            {"trade_id": "dup", "ingest_at": datetime(2023, 5, 3, tzinfo=timezone.utc), **base},
        ]
    )
    dup_audit = audit_duplicate_trades(dup_df)
    assert dup_audit["n_exact_ingestion_duplicates"] == 1
    assert dup_audit["n_divergent_duplicates"] == 0

    canon = canonicalize_trades(dup_df)
    assert canon.height == 1
    assert "fill_id" in canon.columns


def test_adverse_fill_costs_more_than_signal():
    signal = 0.50
    print_px = 0.52
    fill = adverse_fill_price(signal, print_px, "buy", slip_bps=50.0)
    assert fill > signal
    assert fill > print_px

    ts = datetime(2023, 4, 1, tzinfo=timezone.utc)
    prints = pl.DataFrame(
        {
            "token_id": ["tok-a", "tok-a"],
            "traded_at": [
                datetime(2023, 4, 1, 0, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 4, 1, 0, 0, 2, tzinfo=timezone.utc),
            ],
            "price": [0.48, 0.53],
            "side": ["BUY", "BUY"],
        }
    )
    outcome = simulate_outcome_fill(
        prints,
        token_id="tok-a",
        signal_at=datetime(2023, 4, 1, 0, 0, 1, tzinfo=timezone.utc),
        signal_price=0.50,
        side="buy",
        cfg=OutcomeExecConfig(latency_s=0.5, slip_bps=50.0, entry_timeout_s=10.0),
    )
    assert outcome.filled is True
    assert outcome.fill_price is not None
    assert outcome.fill_price >= 0.53
    assert outcome.fee == 0.0  # 2023 historical zero fee


def test_stats_unavailable_without_pvalues():
    result = bh_fdr(None)
    assert result["status"] == "unavailable"
    assert "reason" in result

    dsr = deflated_sharpe_valid(1.5, n_obs=1, n_trials=10)
    assert dsr["status"] == "unavailable"
    assert dsr["dsr"] is None


def test_bootstrap_ci_runs():
    event_pnls = [0.1, -0.05, 0.2, 0.0, 0.15, -0.1, 0.05]
    ci = cluster_bootstrap_ci(event_pnls, n_boot=200, seed=42)
    assert ci["status"] == "ok"
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert 0.0 <= ci["p_positive"] <= 1.0


def test_collector_parse_helpers():
    book = parse_book_message(
        {
            "event_type": "book",
            "asset_id": "123",
            "bids": [{"price": 0.48, "size": 100}],
            "asks": [{"price": 0.52, "size": 50}],
        }
    )
    assert book is not None
    assert book["best_bid"] == pytest.approx(0.48)
    assert book["best_ask"] == pytest.approx(0.52)

    trade = parse_trade_message(
        {"event_type": "last_trade_price", "asset_id": "123", "price": 0.51, "size": 10}
    )
    assert trade is not None
    assert trade["price"] == pytest.approx(0.51)

    assert detect_sequence_gap(10, 12) is True
    assert detect_sequence_gap(10, 11) is False
