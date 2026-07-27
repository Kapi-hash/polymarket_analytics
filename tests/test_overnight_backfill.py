"""Offline unit coverage for overnight research helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from polymarket_analytics.research.overnight_backfill import _market_matches_year, _trade_id
from polymarket_analytics.research.overnight_gate import evaluate_outcome_gate


def test_market_year_filter_accepts_closed_or_end_date() -> None:
    assert _market_matches_year({"closedTime": "2024-01-03T01:02:03Z"}, 2024)
    assert _market_matches_year({"endDate": "2024-12-31"}, 2024)
    assert not _market_matches_year({"closedTime": "2023-12-31T23:59:59Z"}, 2024)
    assert not _market_matches_year({"closedTime": "not-a-date"}, 2024)


def test_trade_id_uses_transaction_and_event_index() -> None:
    assert _trade_id({"transactionHash": "0xabc", "logIndex": 7}, 2) == "0xabc_7"
    assert _trade_id({"tx_hash": "0xdef"}, 9) == "0xdef_9"


def test_gate_blocks_tiny_outcome_frame() -> None:
    frame = pl.DataFrame(
        {
            "trade_id": ["a", "b"],
            "condition_id": ["c1", "c2"],
            "token_won": [True, False],
            "fee_confidence": ["exact", "exact"],
            "traded_at": [
                datetime(2023, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, tzinfo=timezone.utc),
            ],
        }
    )
    result = evaluate_outcome_gate(frame)
    assert result["decision"] == "BLOCKED"
    assert result["evidence"]["n_events"] == 2
