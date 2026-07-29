"""Tests for L2 microstructure research modules."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from polymarket_analytics.l2.compaction import compact_daily_sessions, dedupe_exact_messages
from polymarket_analytics.l2.diagnostics import run_l2_diagnostics
from polymarket_analytics.l2.execution import (
    LATENCY_SCENARIOS_MS,
    BookLevel,
    TakerExecutionConfig,
    maker_fill_bounds,
    round_to_tick,
    simulate_taker_execution,
    walk_book_taker,
)
from polymarket_analytics.l2.feature_catalog import get_feature, list_features
from polymarket_analytics.l2.features import (
    abs_spread,
    classify_cancel_vs_trade,
    multi_level_imbalance,
    ofi_delta,
    refill_rate,
    spread_recovery_time,
    yes_no_complement_deviation,
)
from polymarket_analytics.l2.labels import LABEL_HORIZONS_SEC, compute_forward_labels
from polymarket_analytics.l2.lifecycle import LifecycleConfig, LifecycleState, evaluate_lifecycle_tick
from polymarket_analytics.l2.readiness import assert_not_authorized_from_thin_sample, evaluate_l2_readiness
from polymarket_analytics.l2.reconstruct import apply_messages, microprice
from polymarket_analytics.l2.session import (
    assert_single_session,
    build_session_manifest,
    fingerprint_session,
    sha256_file,
    validate_session_outputs,
    write_session_manifest,
)
from polymarket_analytics.l2.staged_search import run_staged_search
from polymarket_analytics.l2.universe import build_token_universe
from polymarket_analytics.research.logit import logit
from polymarket_analytics.research.validation import purged_walk_forward_folds


def _make_session(tmp: Path, session_id: str, *, duration: float = 3600.0, emitting: list[str] | None = None) -> Path:
    run_dir = tmp / session_id
    raw_dir = run_dir / "raw"
    norm_dir = run_dir / "normalized"
    raw_dir.mkdir(parents=True)
    norm_dir.mkdir(parents=True)
    emitting = emitting or ["token_a", "token_b"]
    raw_path = raw_dir / "session.jsonl"
    raw_path.write_text('{"event_type":"book","asset_id":"token_a"}\n', encoding="utf-8")
    rows = [
        {"event_type": "book", "asset_id": "token_a", "best_bid": 0.48, "best_ask": 0.52,
         "session_id": session_id, "timestamp": "1"},
        {"event_type": "book", "asset_id": "token_b", "best_bid": 0.30, "best_ask": 0.32,
         "session_id": session_id, "timestamp": "2"},
    ]
    pl.DataFrame(rows).write_parquet(norm_dir / "part.parquet")
    health = {
        "duration_sec_actual": duration,
        "emitting_tokens": emitting,
        "session_id": session_id,
    }
    (run_dir / "health.json").write_text(json.dumps(health), encoding="utf-8")
    manifest = build_session_manifest(
        session_id=session_id,
        start_utc="2026-07-01T00:00:00+00:00",
        end_utc="2026-07-01T01:00:00+00:00",
        duration_sec=duration,
        tokens_selected=emitting,
        tokens_emitting=emitting,
        raw_row_count=1,
        normalized_row_count=2,
        run_dir=run_dir,
    )
    write_session_manifest(manifest, run_dir)
    fingerprint_session(run_dir)
    return run_dir


# --- session ---


def test_session_isolation_rejects_mixed_sessions():
    rows = [{"session_id": "a"}, {"session_id": "b"}]
    with pytest.raises(ValueError, match="multiple session_ids"):
        assert_single_session(rows)


def test_session_manifest_and_hashes(tmp_path):
    run_dir = _make_session(tmp_path, "sess1")
    manifest_path = run_dir / "session_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["session_id"] == "sess1"
    assert manifest["artifact_hashes"]
    fp = fingerprint_session(run_dir)
    assert fp["files"]
    assert sha256_file(run_dir / "raw" / "session.jsonl") == fp["files"]["raw/session.jsonl"]


def test_validate_session_outputs_fail_closed(tmp_path):
    run_dir = _make_session(tmp_path, "sess_ok", duration=3600.0)
    result = validate_session_outputs(run_dir, requested_duration_sec=3600.0)
    assert result["ok"] is True

    bad = tmp_path / "bad"
    bad.mkdir()
    bad_result = validate_session_outputs(bad, requested_duration_sec=3600.0)
    assert bad_result["ok"] is False
    assert any("no raw" in e for e in bad_result["errors"])


def test_validate_rejects_stale_abs_paths(tmp_path):
    run_dir = _make_session(tmp_path, "sess_stale")
    health = json.loads((run_dir / "health.json").read_text())
    health["raw_dir"] = "/Users/foo/data/raw"
    (run_dir / "health.json").write_text(json.dumps(health))
    result = validate_session_outputs(run_dir, requested_duration_sec=3600.0, allow_local_abs_paths=False)
    assert result["ok"] is False


def test_validate_rejects_early_end(tmp_path):
    run_dir = _make_session(tmp_path, "sess_early", duration=100.0)
    result = validate_session_outputs(run_dir, requested_duration_sec=3600.0)
    assert result["ok"] is False


# --- universe ---


def test_token_universe_deterministic():
    markets = [
        {"clobTokenIds": [f"t{i}"], "volume": 1000 - i, "category": "crypto",
         "lastTradePrice": 0.3 + (i % 5) * 0.1, "createdAt": "2026-01-01T00:00:00Z",
         "endDate": "2026-12-31T00:00:00Z", "conditionId": f"c{i}", "id": f"e{i}"}
        for i in range(120)
    ]
    u1 = build_token_universe(markets, seed="test", as_of=datetime(2026, 7, 1, tzinfo=timezone.utc))
    u2 = build_token_universe(markets, seed="test", as_of=datetime(2026, 7, 1, tzinfo=timezone.utc))
    assert u1["selected_token_ids"] == u2["selected_token_ids"]
    assert len(u1["core_tokens"]) == 60
    assert len(u1["rotate_tokens"]) == 40


# --- reconstruct ---


def test_book_reconstruction_snapshot_and_delta():
    msgs = [
        {"event_type": "snapshot", "asset_id": "t1", "exchange_time": 1.0, "seq": 1,
         "bids": [{"price": 0.48, "size": 100}], "asks": [{"price": 0.52, "size": 80}]},
        {"event_type": "delta", "asset_id": "t1", "exchange_time": 2.0, "seq": 2,
         "side": "bid", "price": 0.49, "size": 50},
    ]
    records = apply_messages(msgs)
    assert len(records) == 2
    assert records[-1].best_bid == pytest.approx(0.49)
    assert records[-1].mid == pytest.approx((0.49 + 0.52) / 2)


def test_microprice():
    mp = microprice(0.48, 100, 0.52, 80)
    assert mp == pytest.approx((0.52 * 100 + 0.48 * 80) / 180)


def test_gap_invalidates_book():
    msgs = [
        {"event_type": "snapshot", "asset_id": "t1", "exchange_time": 1.0, "seq": 1,
         "bids": [{"price": 0.48, "size": 100}], "asks": [{"price": 0.52, "size": 80}]},
        {"event_type": "delta", "asset_id": "t1", "exchange_time": 2.0, "seq": 5,
         "side": "bid", "price": 0.49, "size": 50},
    ]
    records = apply_messages(msgs)
    assert records[-1].uncertain is True


def test_reconnect_recovery_requires_snapshot():
    msgs = [
        {"event_type": "snapshot", "asset_id": "t1", "exchange_time": 1.0, "seq": 1,
         "bids": [{"price": 0.48, "size": 100}], "asks": [{"price": 0.52, "size": 80}]},
        {"event_type": "reconnect", "asset_id": "t1", "exchange_time": 2.0},
        {"event_type": "delta", "asset_id": "t1", "exchange_time": 3.0, "seq": 2,
         "side": "bid", "price": 0.49, "size": 50},
        {"event_type": "snapshot", "asset_id": "t1", "exchange_time": 4.0, "seq": 10,
         "bids": [{"price": 0.49, "size": 50}], "asks": [{"price": 0.52, "size": 80}]},
    ]
    records = apply_messages(msgs)
    uncertain = [r for r in records if r.uncertain]
    assert len(uncertain) >= 1
    assert records[-1].uncertain is False


def test_timestamp_inversion_detected():
    msgs = [
        {"event_type": "snapshot", "asset_id": "t1", "exchange_time": 5.0, "seq": 1,
         "bids": [{"price": 0.48, "size": 100}], "asks": [{"price": 0.52, "size": 80}]},
        {"event_type": "snapshot", "asset_id": "t1", "exchange_time": 3.0, "seq": 2,
         "bids": [{"price": 0.48, "size": 100}], "asks": [{"price": 0.52, "size": 80}]},
    ]
    records = apply_messages(msgs)
    assert len(records) == 2


# --- features ---


def test_multi_level_imbalance():
    imb = multi_level_imbalance([10, 5], [8, 4])
    assert imb == pytest.approx((15 - 12) / (15 + 12))


def test_ofi_delta():
    assert ofi_delta(12, 8, 10, 8) == pytest.approx(2.0)


def test_cancel_refill_classification():
    assert classify_cancel_vs_trade(100, 80, 25) == "trade"
    assert classify_cancel_vs_trade(100, 90, 0) == "cancel"
    assert classify_cancel_vs_trade(100, 110, 0) == "add"


def test_resilience_spread_recovery():
    spreads = [0.01, 0.01, 0.05, 0.04, 0.02, 0.01]
    t = spread_recovery_time(spreads, 2, baseline=0.01)
    assert t == pytest.approx(3.0)


def test_logit_near_boundaries():
    z_low = logit(0.01)
    z_high = logit(0.99)
    assert z_low < z_high
    assert abs_spread(0.48, 0.52) == pytest.approx(0.04)


def test_yes_no_complement():
    assert yes_no_complement_deviation(0.48, 0.50) == pytest.approx(-0.02)


# --- execution ---


def test_book_walking_partial_fills():
    asks = [BookLevel(0.52, 30), BookLevel(0.53, 50)]
    filled, avg, consumed, residual = walk_book_taker("buy", 50, asks, partial_fill=True)
    assert filled == pytest.approx(50)
    assert consumed == 2
    assert residual == pytest.approx(0)


def test_taker_fees_and_latency():
    asks = [BookLevel(0.52, 100)]
    bids = [BookLevel(0.48, 100)]
    cfg = TakerExecutionConfig(latency_ms=LATENCY_SCENARIOS_MS[0], fee_category="crypto")
    fill = simulate_taker_execution(
        "buy", 10, bids=bids, asks=asks, mid_at_entry=0.50, mid_after=0.49, cfg=cfg,
    )
    assert fill.filled_size == pytest.approx(10)
    assert fill.fee > 0
    assert fill.adverse_selection == pytest.approx(0.52 - 0.49)


def test_reject_stale_uncertain_book():
    fill = simulate_taker_execution(
        "buy", 10,
        bids=[BookLevel(0.48, 100)], asks=[BookLevel(0.52, 100)],
        mid_at_entry=0.50, book_uncertain=True,
    )
    assert fill.rejected is True


def test_maker_bounds_conservative():
    bounds = maker_fill_bounds(queue_ahead=50, trade_through=30, visible_depth=100)
    assert bounds.lower_prob <= bounds.base_prob <= bounds.upper_prob
    assert bounds.lower_prob == 0.0  # touch_does_not_fill


def test_tick_rounding():
    assert round_to_tick(0.523, tick=0.01) == pytest.approx(0.52)


# --- labels ---


def test_forward_labels_and_gap_skip():
    series = [
        {"time": 10, "mid": 0.51},
        {"time": 20, "mid": 0.53},
        {"time": 40, "mid": 0.55},
    ]
    labels = compute_forward_labels(
        entry_time=5, entry_mid=0.50, entry_side="long", series=series,
        horizons_sec=[5, 30], gaps=[(15, 25)],
    )
    assert labels[0].skipped is False
    assert labels[1].skipped is True
    assert labels[1].skip_reason == "gap"


def test_forward_label_horizons_default():
    assert len(LABEL_HORIZONS_SEC) == 7


# --- lifecycle ---


def test_gap_aware_lifecycle_forced_exit():
    state = LifecycleState(entry_price=0.50, entry_time=0.0)
    cfg = LifecycleConfig(allow_gap_hold=False)
    out = evaluate_lifecycle_tick(state, mark=0.51, now=1.0, in_gap=True, cfg=cfg)
    assert out.exit_reason == "gap_forced"


def test_lifecycle_take_profit():
    state = LifecycleState(entry_price=0.50, entry_time=0.0)
    out = evaluate_lifecycle_tick(state, mark=0.56, now=10.0, cfg=LifecycleConfig(take_profit_pct=0.10))
    assert out.exit_reason == "take_profit"


# --- compaction ---


def test_dedupe_exact_messages():
    rows = [{"a": 1}, {"a": 1}, {"a": 2}]
    deduped, dupes, div = dedupe_exact_messages(rows)
    assert dupes == 1
    assert len(deduped) == 2


def test_daily_compaction(tmp_path):
    s1 = _make_session(tmp_path, "d1")
    out = tmp_path / "compact"
    result = compact_daily_sessions([s1], utc_date="2026-07-01", out_dir=out, requested_duration_sec=3600.0)
    assert result["report"]["total_rows"] >= 1
    assert (out / "artifact_index.json").is_file()


# --- readiness ---


def test_readiness_rejects_thin_data():
    cov = {"calendar_days": 1, "emitting_tokens": 5, "independent_events": 2,
           "utc_dayparts": ["morning"], "probability_buckets": ["mid"],
           "ttr_regimes": ["lt_1w"], "pit_metadata_ok": False, "fees_valid": False,
           "splits": {"train": [], "val": [], "locked_test": []}}
    result = evaluate_l2_readiness(cov)
    assert result["decision"] == "NOT_READY"
    assert result["research_authorized"] is False
    assert result["diagnostic_only"] is True


def test_assert_not_authorized_from_thin_sample():
    with pytest.raises(PermissionError):
        assert_not_authorized_from_thin_sample(5.0)


# --- staged search & diagnostics ---


def test_staged_search_blocks_definitive_on_thin_data():
    cov = {"calendar_days": 1, "collection_hours": 5, "emitting_tokens": 10,
           "independent_events": 5, "utc_dayparts": ["a"], "probability_buckets": ["mid"],
           "ttr_regimes": ["lt_1w"], "pit_metadata_ok": False, "fees_valid": False,
           "splits": {"train": [], "val": [], "locked_test": []}}
    r = run_staged_search(stage=3, coverage=cov)
    assert r.status == "blocked"


def test_staged_search_allows_diagnostic_stages():
    cov = {"calendar_days": 0.2, "collection_hours": 5, "emitting_tokens": 3,
           "independent_events": 1, "utc_dayparts": ["a"], "probability_buckets": ["mid"],
           "ttr_regimes": ["lt_1w"], "pit_metadata_ok": False, "fees_valid": False,
           "splits": {"train": [], "val": [], "locked_test": []}}
    r = run_staged_search(stage=1, coverage=cov)
    assert r.status == "diagnostic_only"
    assert r.research_authorized is False


def test_diagnostics_never_authorizes():
    diag = run_l2_diagnostics()
    assert diag["research_authorized"] is False
    assert diag["paper_authorized"] is False
    assert diag["non_definitive"] is True


# --- validation stubs ---


def test_purged_wf_split_stub():
    folds = purged_walk_forward_folds(
        start="2026-01-01", end="2026-04-01", n_folds=2, train_days=30, test_days=7, embargo_days=2,
    )
    assert len(folds) >= 1
    assert folds[0].embargo_start <= folds[0].test_start


def test_locked_test_isolation_stub():
    cov = {"calendar_days": 10, "emitting_tokens": 60, "independent_events": 25,
           "utc_dayparts": ["morning", "evening"], "probability_buckets": ["low", "mid"],
           "ttr_regimes": ["lt_1d", "lt_1w"], "pit_metadata_ok": True, "fees_valid": True,
           "splits": {"train": ["a"], "val": ["b"], "locked_test": ["c"]}}
    result = evaluate_l2_readiness(cov)
    assert result["decision"] == "RESEARCH_READY"


# --- catalog ---


def test_feature_catalog_blocked_cross_market():
    entry = get_feature("cross_market_lead_lag")
    assert entry.status == "blocked"
    implemented = list_features(status="implemented")
    assert len(implemented) > 20


def test_refill_rate():
    assert refill_rate(100, 150, 10.0) == pytest.approx(0.5 / 10.0)
