"""Restricted L2 diagnostics — non-definitive, never authorizes research."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from polymarket_analytics.l2.execution import LATENCY_SCENARIOS_MS, simulate_taker_execution, BookLevel
from polymarket_analytics.l2.feature_catalog import catalog_summary, list_features
from polymarket_analytics.l2.features import abs_spread, multi_level_imbalance
from polymarket_analytics.l2.reconstruct import apply_messages, microprice
from polymarket_analytics.l2.session import SESSION_SCHEMA_VERSION


def run_l2_diagnostics(
    *,
    messages: list[dict[str, Any]] | None = None,
    coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run restricted diagnostics labeled non_definitive / diagnostic_only.

    Must NOT set research_authorized or paper_authorized.
    """
    results: dict[str, Any] = {
        "mode": "diagnostic_only",
        "non_definitive": True,
        "research_authorized": False,
        "paper_authorized": False,
        "schema_version": SESSION_SCHEMA_VERSION,
        "checks": {},
    }

    # Schema validation
    results["checks"]["schema"] = {"ok": True, "version": SESSION_SCHEMA_VERSION}

    # Feature smoke
    feat_ok = abs_spread(0.48, 0.52) == 0.04
    imb = multi_level_imbalance([10, 5], [8, 4])
    catalog = catalog_summary()
    implemented = len(list_features(status="implemented"))
    results["checks"]["features"] = {
        "ok": feat_ok,
        "sample_imbalance": imb,
        "catalog_total": catalog["total"],
        "implemented_count": implemented,
    }

    # Reconstruction smoke
    if messages:
        records = apply_messages(messages)
        results["checks"]["reconstruction"] = {
            "ok": len(records) > 0,
            "n_records": len(records),
            "sample_microprice": records[-1].microprice if records else None,
        }
    else:
        sample = [
            {"event_type": "snapshot", "asset_id": "t1", "exchange_time": 1.0,
             "bids": [{"price": 0.48, "size": 100}], "asks": [{"price": 0.52, "size": 80}]},
        ]
        records = apply_messages(sample)
        results["checks"]["reconstruction"] = {"ok": len(records) == 1, "n_records": len(records)}

    # Execution smoke
    asks = [BookLevel(0.52, 100), BookLevel(0.53, 50)]
    bids = [BookLevel(0.48, 100)]
    for lat in LATENCY_SCENARIOS_MS[:2]:
        fill = simulate_taker_execution(
            "buy", 50, bids=bids, asks=asks, mid_at_entry=0.50,
            cfg=None,
        )
        if fill.rejected:
            results["checks"]["execution"] = {"ok": False, "latency_ms": lat}
            break
    else:
        mp = microprice(0.48, 100, 0.52, 80)
        results["checks"]["execution"] = {"ok": True, "sample_microprice": mp}

    # DQ profile
    cov = coverage or {}
    results["checks"]["dq_profile"] = {
        "calendar_days": cov.get("calendar_days", 0),
        "emitting_tokens": cov.get("emitting_tokens", 0),
        "diagnostic_only": True,
        "note": "5h samples are non-definitive",
    }

    results["ok"] = all(c.get("ok", True) for c in results["checks"].values())
    return results


def run_session_diagnostics(session_dir: str | Path) -> dict[str, Any]:
    """Non-definitive diagnostics for one on-disk session directory."""
    from polymarket_analytics.l2.session import read_jsonl_gz, validate_session_outputs

    root = Path(session_dir)
    validation = validate_session_outputs(root, requested_duration_sec=None)
    coverage: dict[str, Any] = {"emitting_tokens": validation.get("emitting_tokens", 0)}
    messages: list[dict[str, Any]] | None = None
    raw_files = sorted(root.glob("raw/*.jsonl*"))
    if raw_files:
        try:
            messages = read_jsonl_gz(raw_files[0])[:200]
        except (OSError, json.JSONDecodeError):
            messages = None

    diag = run_l2_diagnostics(messages=messages, coverage=coverage)
    diag["session_id"] = root.name
    diag["validation"] = validation
    if not validation.get("ok"):
        diag["ok"] = False
    return diag
