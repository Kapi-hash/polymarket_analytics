"""Daily compaction of L2 session artifacts."""

from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from polymarket_analytics.l2.session import validate_session_outputs


def _session_utc_date(session_dir: Path) -> str | None:
    manifest = session_dir / "session_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if data.get("utc_date"):
                return data.get("utc_date")
            extra = data.get("extra") or {}
            if extra.get("utc_date"):
                return extra.get("utc_date")
            start = data.get("start_utc") or data.get("started_at")
            if start:
                return str(start)[:10]
        except json.JSONDecodeError:
            pass
    return None


def find_sessions_for_date(sessions_dir: Path, utc_date: str) -> list[Path]:
    if not sessions_dir.exists():
        return []
    out: list[Path] = []
    for child in sorted(sessions_dir.iterdir()):
        if not child.is_dir():
            continue
        if _session_utc_date(child) == utc_date:
            out.append(child)
    return out


def compact_day(
    utc_date: str,
    sessions_dir: Path | str,
    out_dir: Path | str,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """
    Compact all sessions for ``utc_date`` into a daily bundle + coverage report.

    Fail closed when any session fails validation or parquet schema checks.
    """
    sessions_dir = Path(sessions_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = find_sessions_for_date(sessions_dir, utc_date)
    if not sessions:
        return {
            "ok": False,
            "utc_date": utc_date,
            "error": f"no sessions found for {utc_date} under {sessions_dir}",
            "sessions": [],
        }

    session_reports: list[dict[str, Any]] = []
    total_norm_rows = 0
    total_raw_bytes = 0
    emitting_union: set[str] = set()

    for session in sessions:
        if validate:
            v = validate_session_outputs(session, requested_duration_sec=None)
            if not v.get("ok"):
                return {
                    "ok": False,
                    "utc_date": utc_date,
                    "error": "session validation failed",
                    "failed_session": session.name,
                    "validation": v,
                }
            session_reports.append(v)
        else:
            session_reports.append({"session_id": session.name})

        for raw in session.glob("raw/*"):
            if raw.is_file():
                total_raw_bytes += raw.stat().st_size
        for pq in session.glob("normalized/**/*.parquet"):
            try:
                total_norm_rows += pl.read_parquet(pq).height
            except Exception as exc:  # noqa: BLE001
                return {
                    "ok": False,
                    "utc_date": utc_date,
                    "error": f"corrupt parquet {pq}: {exc}",
                }
        health_path = session / "health.json"
        if not health_path.exists():
            health_path = session / "reports" / "collector_health.json"
        if health_path.exists():
            try:
                health = json.loads(health_path.read_text(encoding="utf-8"))
                for tok in health.get("emitting_tokens") or []:
                    emitting_union.add(str(tok))
            except json.JSONDecodeError:
                pass

    bundle_name = f"l2-daily-{utc_date}.tar.gz"
    bundle_path = out_dir / bundle_name
    with tarfile.open(bundle_path, "w:gz") as tar:
        for session in sessions:
            tar.add(session, arcname=f"sessions/{session.name}")

    bundle_bytes = bundle_path.stat().st_size
    sha256 = hashlib.sha256(bundle_path.read_bytes()).hexdigest()

    coverage = {
        "utc_date": utc_date,
        "n_sessions": len(sessions),
        "session_ids": [s.name for s in sessions],
        "total_normalized_rows": total_norm_rows,
        "total_raw_bytes": total_raw_bytes,
        "bundle_bytes": bundle_bytes,
        "bundle_sha256": sha256,
        "n_emitting_tokens_union": len(emitting_union),
        "estimated_storage_mb": round((total_raw_bytes + bundle_bytes) / (1024 * 1024), 2),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sessions": session_reports,
    }

    coverage_path = out_dir / f"coverage_{utc_date}.json"
    coverage_path.write_text(json.dumps(coverage, indent=2, default=str), encoding="utf-8")

    storage_warn = coverage["estimated_storage_mb"] > 5000
    return {
        "ok": True,
        "utc_date": utc_date,
        "bundle": str(bundle_path),
        "coverage_json": str(coverage_path),
        "coverage": coverage,
        "storage_warning": storage_warn,
        "storage_warning_message": (
            f"estimated storage {coverage['estimated_storage_mb']} MB exceeds 5 GB threshold"
            if storage_warn
            else None
        ),
    }
