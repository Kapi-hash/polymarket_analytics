"""Continuous L2 microstructure research modules."""

from __future__ import annotations

from polymarket_analytics.l2.compaction import compact_daily_sessions, dedupe_exact_messages
from polymarket_analytics.l2.diagnostics import run_l2_diagnostics
from polymarket_analytics.l2.execution import (
    LATENCY_SCENARIOS_MS,
    MakerFillBounds,
    TakerFillResult,
    maker_fill_bounds,
    simulate_taker_execution,
    walk_book_taker,
)
from polymarket_analytics.l2.feature_catalog import FEATURE_CATALOG, catalog_summary, get_feature, list_features
from polymarket_analytics.l2.features import (
    abs_spread,
    multi_level_imbalance,
    ofi_delta,
    signed_trade_imbalance,
)
from polymarket_analytics.l2.labels import LABEL_HORIZONS_SEC, ForwardLabel, compute_forward_labels
from polymarket_analytics.l2.lifecycle import LifecycleConfig, LifecycleState, evaluate_lifecycle_tick
from polymarket_analytics.l2.readiness import (
    MIN_CALENDAR_DAYS,
    assert_not_authorized_from_thin_sample,
    evaluate_l2_readiness,
)
from polymarket_analytics.l2.reconstruct import BookSnapshotRecord, OrderBook, apply_messages, microprice
from polymarket_analytics.l2.session import (
    SESSION_SCHEMA_VERSION,
    assert_single_session,
    build_session_manifest,
    fingerprint_session,
    sha256_file,
    validate_session_outputs,
    write_session_manifest,
)
from polymarket_analytics.l2.staged_search import run_staged_search
from polymarket_analytics.l2.universe import build_token_universe, discover_markets_for_universe

__all__ = [
    "SESSION_SCHEMA_VERSION",
    "MIN_CALENDAR_DAYS",
    "LATENCY_SCENARIOS_MS",
    "LABEL_HORIZONS_SEC",
    "FEATURE_CATALOG",
    "BookSnapshotRecord",
    "ForwardLabel",
    "LifecycleConfig",
    "LifecycleState",
    "MakerFillBounds",
    "OrderBook",
    "TakerFillResult",
    "apply_messages",
    "assert_not_authorized_from_thin_sample",
    "assert_single_session",
    "build_session_manifest",
    "build_token_universe",
    "catalog_summary",
    "compact_daily_sessions",
    "compute_forward_labels",
    "dedupe_exact_messages",
    "discover_markets_for_universe",
    "evaluate_l2_readiness",
    "evaluate_lifecycle_tick",
    "fingerprint_session",
    "get_feature",
    "list_features",
    "maker_fill_bounds",
    "microprice",
    "run_l2_diagnostics",
    "run_staged_search",
    "sha256_file",
    "simulate_taker_execution",
    "validate_session_outputs",
    "walk_book_taker",
    "write_session_manifest",
]
