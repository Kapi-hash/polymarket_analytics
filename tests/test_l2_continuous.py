"""Tests for L2 continuous collection infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from polymarket_analytics.collectors.book_collector import NORMALIZED_SCHEMA, write_normalized_rows
from polymarket_analytics.l2.compact import find_sessions_for_date
from polymarket_analytics.l2.readiness import evaluate_readiness
from polymarket_analytics.l2.session import (
    build_session_manifest,
    fingerprint_session,
    reject_fixture_paths,
    validate_session_outputs,
    write_gap_report,
    write_metadata,
    write_session_manifest,
)
from polymarket_analytics.l2.universe import build_diversified_universe, build_top_universe


def test_reject_fixture_paths() -> None:
    with pytest.raises(ValueError, match="fixture"):
        reject_fixture_paths("tests/fixtures/books_smoke/out")


def test_session_manifest_and_validation(tmp_path: Path) -> None:
    session = tmp_path / "sess1"
    (session / "raw").mkdir(parents=True)
    (session / "normalized" / "utc_date=2026-07-28").mkdir(parents=True)
    raw = session / "raw" / "session_sess1.jsonl.gz"
    raw.write_bytes(b"\x1f\x8b")  # placeholder; validation checks existence
    norm = session / "normalized" / "utc_date=2026-07-28" / "part.parquet"
    rows = [
        {
            "event_type": "book",
            "asset_id": "123",
            "best_bid": 0.4,
            "best_ask": 0.6,
            "price": None,
            "size": None,
            "side": None,
            "timestamp": "1",
            "received_at": "2026-07-28T00:00:00+00:00",
            "record_type": "book",
            "raw_keys": ["asset_id"],
            "session_id": "sess1",
            "utc_date": "2026-07-28",
            "parse_ok": True,
        }
    ]
    write_normalized_rows(rows, norm)
    write_metadata(session, token_ids=["123"], market_meta=[])
    write_gap_report(session, session_id="sess1", health={"gaps_detected": 0, "reconnects": 0})
    health = {
        "ok": True,
        "emitting_tokens": ["123"],
        "duration_sec_actual": 100.0,
        "health": {"emitting_tokens": ["123"], "raw_rows": 1, "normalized_rows": 1},
    }
    (session / "health.json").write_text(json.dumps(health), encoding="utf-8")
    manifest = build_session_manifest(
        session_id="sess1",
        start_utc="2026-07-28T00:00:00+00:00",
        end_utc="2026-07-28T00:01:40+00:00",
        duration_sec=100.0,
        tokens_selected=["123"],
        tokens_emitting=["123"],
        raw_row_count=1,
        normalized_row_count=1,
        run_dir=session,
        extra={"utc_date": "2026-07-28"},
    )
    write_session_manifest(manifest, session)
    fingerprint_session(session)
    result = validate_session_outputs(session, requested_duration_sec=100.0)
    assert result["ok"] is True
    assert result["emitting_tokens"] == 1


def test_readiness_gate(tmp_path: Path) -> None:
    cov_path = tmp_path / "cov.json"
    cov_path.write_text(
        json.dumps(
            {
                "n_sessions": 8,
                "n_emitting_tokens_union": 60,
                "total_normalized_rows": 250_000,
                "session_ids": ["a", "b"],
            }
        ),
        encoding="utf-8",
    )
    not_ready = evaluate_readiness(tmp_path / "missing.json")
    assert not_ready["ready"] is False
    result = evaluate_readiness(cov_path)
    assert result["ok"] is True
    assert "ready" in result


def test_universe_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_markets = [
        {
            "clobTokenIds": json.dumps([f"tok{i}"]),
            "conditionId": f"c{i}",
            "volume24hr": 1000 - i,
            "category": "crypto",
        }
        for i in range(150)
    ]

    def _fake_discover(limit: int = 500):
        return fake_markets

    monkeypatch.setattr(
        "polymarket_analytics.l2.universe.discover_markets_for_universe",
        _fake_discover,
    )
    div = build_diversified_universe(n_core=60, n_rotate=40, seed=7)
    assert len(div["selected_tokens"]) == 100
    assert div["mode"] == "diversified"

    monkeypatch.setattr(
        "polymarket_analytics.collectors.book_collector.discover_active_token_ids",
        lambda n: [f"top{i}" for i in range(n)],
    )
    top = build_top_universe(50)
    assert top["mode"] == "top"
    assert len(top["selected_tokens"]) == 50


def test_find_sessions_for_date(tmp_path: Path) -> None:
    s1 = tmp_path / "a"
    s1.mkdir()
    (s1 / "session_manifest.json").write_text(
        json.dumps({"extra": {"utc_date": "2026-07-28"}}), encoding="utf-8"
    )
    found = find_sessions_for_date(tmp_path, "2026-07-28")
    assert found == [s1]


def test_normalized_schema_has_string_ids() -> None:
    assert NORMALIZED_SCHEMA["asset_id"] == pl.Utf8
    assert NORMALIZED_SCHEMA["session_id"] == pl.Utf8
