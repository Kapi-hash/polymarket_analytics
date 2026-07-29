"""Daily L2 compaction with dedupe and artifact indexing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from polymarket_analytics.l2.session import (
    SESSION_SCHEMA_VERSION,
    assert_single_session,
    fingerprint_session,
    sha256_file,
    validate_session_outputs,
)

ACTIONS_STORAGE_WARN_GB: float = 45.0
ACTIONS_STORAGE_LIMIT_GB: float = 50.0


@dataclass
class CompactionReport:
    utc_date: str
    sessions_processed: int
    sessions_rejected: int
    exact_duplicates_removed: int
    divergent_preserved: int
    total_rows: int
    emitting_tokens: int
    artifacts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    storage_estimate_gb: float = 0.0
    storage_warning: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def dedupe_exact_messages(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    """Dedupe only exact message copies; preserve divergent rows."""
    seen: dict[str, dict[str, Any]] = {}
    dupes = 0
    divergent = 0
    for row in rows:
        key = hashlib.sha256(_canonical_json(row).encode()).hexdigest()
        if key in seen:
            if _canonical_json(seen[key]) != _canonical_json(row):
                divergent += 1
                seen[f"{key}:{divergent}"] = row
            else:
                dupes += 1
        else:
            seen[key] = row
    return list(seen.values()), dupes, divergent


def _estimate_storage_gb(paths: list[Path]) -> float:
    total = sum(p.stat().st_size for p in paths if p.is_file())
    return total / (1024**3)


def compact_daily_sessions(
    session_dirs: list[Path | str],
    *,
    utc_date: str,
    out_dir: Path | str,
    requested_duration_sec: float = 86400.0,
) -> dict[str, Any]:
    """Compact multiple sessions into daily artifact with validation."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = CompactionReport(utc_date=utc_date, sessions_processed=0, sessions_rejected=0,
                              exact_duplicates_removed=0, divergent_preserved=0,
                              total_rows=0, emitting_tokens=0)

    all_rows: list[dict[str, Any]] = []
    session_ids: set[str] = set()
    file_paths: list[Path] = []

    for sdir in session_dirs:
        root = Path(sdir)
        validation = validate_session_outputs(root, requested_duration_sec=requested_duration_sec)
        if not validation["ok"]:
            report.sessions_rejected += 1
            report.errors.extend([f"{root.name}: {e}" for e in validation["errors"]])
            continue
        report.sessions_processed += 1
        manifest_path = root / "session_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sid = manifest.get("session_id", root.name)
        if sid in session_ids:
            report.sessions_rejected += 1
            report.errors.append(f"overlapping session_id: {sid}")
            continue
        session_ids.add(sid)

        norm_dir = root / "normalized"
        for fp in sorted(norm_dir.rglob("*.parquet")):
            file_paths.append(fp)
            df = pl.read_parquet(fp)
            if "session_id" in df.columns:
                try:
                    assert_single_session(df)
                except ValueError as exc:
                    report.sessions_rejected += 1
                    report.errors.append(str(exc))
                    continue
            for row in df.to_dicts():
                row["utc_date"] = utc_date
                all_rows.append(row)

    deduped, dupes, divergent = dedupe_exact_messages(all_rows)
    report.exact_duplicates_removed = dupes
    report.divergent_preserved = divergent
    report.total_rows = len(deduped)

    if deduped:
        df = pl.DataFrame(deduped)
        out_parquet = out / f"compacted_{utc_date}.parquet"
        df.write_parquet(out_parquet, compression="snappy")
        report.artifacts.append(str(out_parquet.name))
        file_paths.append(out_parquet)
        if "asset_id" in df.columns:
            report.emitting_tokens = df["asset_id"].n_unique()
        elif "token_id" in df.columns:
            report.emitting_tokens = df["token_id"].n_unique()

    report.storage_estimate_gb = _estimate_storage_gb(file_paths)
    report.storage_warning = report.storage_estimate_gb >= ACTIONS_STORAGE_WARN_GB

    index = {
        "utc_date": utc_date,
        "schema_version": SESSION_SCHEMA_VERSION,
        "sessions": sorted(session_ids),
        "artifacts": report.artifacts,
        "coverage": {
            "total_rows": report.total_rows,
            "emitting_tokens": report.emitting_tokens,
            "exact_duplicates_removed": report.exact_duplicates_removed,
            "divergent_preserved": report.divergent_preserved,
        },
        "storage_estimate_gb": report.storage_estimate_gb,
        "storage_warning": report.storage_warning,
    }
    index_path = out / "artifact_index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    report.artifacts.append(index_path.name)

    coverage_path = out / "coverage_report.json"
    coverage_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return {"report": report.to_dict(), "index": index, "ok": report.sessions_rejected == 0 or report.sessions_processed > 0}
