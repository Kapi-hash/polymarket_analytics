"""Session layout, manifests, validation, and fingerprinting for L2 book capture."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterable

import polars as pl

SESSION_SCHEMA_VERSION: Final[str] = "l2-session-v1"
SESSION_ROOT: Final[str] = "data/books/sessions"

SESSION_SUBDIRS: Final[tuple[str, ...]] = (
    "raw",
    "normalized",
    "snapshots",
    "metadata",
    "logs",
)

SESSION_ARTIFACTS: Final[tuple[str, ...]] = (
    "health.json",
    "gap_report.json",
    "session_manifest.json",
    "fingerprints.json",
)

_STALE_ABS_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"/Users/"),
    re.compile(r"C:\\", re.IGNORECASE),
    re.compile(r"tests/fixtures"),
)


def utc_date_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


FIXTURE_PATH_MARKERS: Final[tuple[str, ...]] = (
    "fixtures",
    "books_smoke",
    "tests/fixtures",
    "sample_trades",
    "sample_markets",
)


def reject_fixture_paths(*paths: Path | str) -> None:
    """Fail closed when output paths resemble packaging fixtures."""
    for raw in paths:
        text = str(raw).replace("\\", "/").lower()
        for marker in FIXTURE_PATH_MARKERS:
            if marker in text:
                raise ValueError(f"output path looks like a fixture: {raw} (marker={marker})")


def write_metadata(
    run_dir: Path | str,
    *,
    token_ids: list[str],
    market_meta: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    root = Path(run_dir)
    meta_dir = root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = meta_dir / "tokens.json"
    tokens_path.write_text(
        json.dumps({"token_ids": [str(t) for t in token_ids], "count": len(token_ids)}, indent=2),
        encoding="utf-8",
    )
    markets_path = meta_dir / "markets.json"
    markets_path.write_text(
        json.dumps({"markets": market_meta or [], "count": len(market_meta or [])}, indent=2, default=str),
        encoding="utf-8",
    )
    return {"tokens": tokens_path, "markets": markets_path}


def write_gap_report(
    run_dir: Path | str,
    *,
    session_id: str,
    health: dict[str, Any],
    gap_events: list[dict[str, Any]] | None = None,
    reconnect_events: list[dict[str, Any]] | None = None,
) -> Path:
    report = {
        "session_id": session_id,
        "gaps_detected": health.get("gaps_detected", 0),
        "reconnects": health.get("reconnects", 0),
        "gap_events": gap_events or [],
        "reconnect_events": reconnect_events or [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out = Path(run_dir) / "gap_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return out


def sha256_file(path: Path | str) -> str:
    """Return hex SHA-256 digest of file contents."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def session_dir(session_id: str, *, data_root: Path | str = "data") -> Path:
    """Return canonical session directory path."""
    return Path(data_root) / "books" / "sessions" / session_id


def ensure_session_layout(run_dir: Path | str) -> dict[str, Path]:
    """Create expected session subdirectories and return path map."""
    root = Path(run_dir)
    paths: dict[str, Path] = {"root": root}
    for name in SESSION_SUBDIRS:
        sub = root / name
        sub.mkdir(parents=True, exist_ok=True)
        paths[name] = sub
    return paths


def write_jsonl_gz(rows: Iterable[dict[str, Any]], path: Path | str) -> None:
    """Write JSONL rows to a gzip-compressed file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")


def read_jsonl_gz(path: Path | str) -> list[dict[str, Any]]:
    """Read JSONL from plain or gzip-compressed file."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" or str(p).endswith(".jsonl.gz") else open
    rows: list[dict[str, Any]] = []
    with opener(p, "rt", encoding="utf-8") as fh:  # type: ignore[arg-type]
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalized_partition_path(
    base: Path | str,
    *,
    utc_date: str,
    token_id: str,
) -> Path:
    """Partition path for compacted normalized frames."""
    return Path(base) / f"utc_date={utc_date}" / f"token_id={token_id}"


@dataclass
class SessionManifest:
    session_id: str
    start_utc: str
    end_utc: str | None
    duration_sec: float
    tokens_selected: list[str]
    tokens_emitting: list[str]
    raw_row_count: int
    normalized_row_count: int
    artifacts: dict[str, str]
    artifact_hashes: dict[str, str]
    schema_version: str = SESSION_SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_session_manifest(
    *,
    session_id: str,
    start_utc: datetime | str,
    end_utc: datetime | str | None,
    duration_sec: float,
    tokens_selected: list[str],
    tokens_emitting: list[str],
    raw_row_count: int,
    normalized_row_count: int,
    run_dir: Path | str,
    schema_version: str = SESSION_SCHEMA_VERSION,
    extra: dict[str, Any] | None = None,
) -> SessionManifest:
    """Build session manifest with relative artifact paths and content hashes."""
    root = Path(run_dir)

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(root))
        except ValueError:
            return str(p)

    artifacts: dict[str, str] = {}
    hashes: dict[str, str] = {}

    for sub in SESSION_SUBDIRS:
        sub_path = root / sub
        if sub_path.exists():
            for fp in sorted(sub_path.rglob("*")):
                if fp.is_file():
                    key = _rel(fp)
                    artifacts[key] = key
                    hashes[key] = sha256_file(fp)

    for name in SESSION_ARTIFACTS:
        if name in {"health.json", "validation.json", "collector_health.json"}:
            continue
        fp = root / name
        if fp.is_file():
            artifacts[name] = name
            hashes[name] = sha256_file(fp)

    if isinstance(start_utc, datetime):
        start_s = start_utc.astimezone(timezone.utc).isoformat()
    else:
        start_s = str(start_utc)
    if end_utc is None:
        end_s = None
    elif isinstance(end_utc, datetime):
        end_s = end_utc.astimezone(timezone.utc).isoformat()
    else:
        end_s = str(end_utc)

    return SessionManifest(
        session_id=session_id,
        start_utc=start_s,
        end_utc=end_s,
        duration_sec=float(duration_sec),
        tokens_selected=list(tokens_selected),
        tokens_emitting=list(tokens_emitting),
        raw_row_count=int(raw_row_count),
        normalized_row_count=int(normalized_row_count),
        artifacts=artifacts,
        artifact_hashes=hashes,
        schema_version=schema_version,
        extra=dict(extra or {}),
    )


def write_session_manifest(manifest: SessionManifest, run_dir: Path | str) -> Path:
    """Persist session manifest JSON."""
    path = Path(run_dir) / "session_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def _contains_stale_abs_path(text: str) -> bool:
    return any(p.search(text) for p in _STALE_ABS_PATTERNS)


def _scan_paths_for_stale(obj: Any) -> list[str]:
    hits: list[str] = []
    if isinstance(obj, str):
        if _contains_stale_abs_path(obj):
            hits.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            hits.extend(_scan_paths_for_stale(v))
    elif isinstance(obj, list):
        for v in obj:
            hits.extend(_scan_paths_for_stale(v))
    return hits


def assert_single_session(rows: list[dict[str, Any]] | pl.DataFrame, *, session_col: str = "session_id") -> str:
    """Reject mixing sessions without provenance."""
    if isinstance(rows, pl.DataFrame):
        if session_col not in rows.columns:
            raise ValueError(f"missing {session_col} column — cannot verify session isolation")
        ids = rows[session_col].drop_nulls().unique().to_list()
    else:
        ids = sorted({str(r[session_col]) for r in rows if r.get(session_col) is not None})
        if any(session_col not in r for r in rows):
            raise ValueError(f"missing {session_col} on some rows — cannot verify session isolation")
    if len(ids) != 1:
        raise ValueError(f"multiple session_ids in batch: {ids}")
    return str(ids[0])


def validate_session_outputs(
    run_dir: Path | str,
    *,
    requested_duration_sec: float | None = None,
    allow_local_abs_paths: bool = False,
) -> dict[str, Any]:
    """Fail-closed validation of session directory outputs."""
    root = Path(run_dir)
    errors: list[str] = []
    warnings: list[str] = []

    raw_dir = root / "raw"
    norm_dir = root / "normalized"
    raw_files = list(raw_dir.rglob("*")) if raw_dir.exists() else []
    raw_files = [f for f in raw_files if f.is_file()]
    norm_files = list(norm_dir.rglob("*.parquet")) if norm_dir.exists() else []

    if not raw_files:
        errors.append("no raw artifacts")
    if not norm_files:
        errors.append("no normalized artifacts")

    health_path = root / "health.json"
    manifest_path = root / "session_manifest.json"
    fp_path = root / "fingerprints.json"

    for required in ("session_manifest.json", "fingerprints.json"):
        if not (root / required).is_file():
            errors.append(f"missing {required}")

    health: dict[str, Any] = {}
    if health_path.is_file():
        health = json.loads(health_path.read_text(encoding="utf-8"))
    elif (root / "reports" / "collector_health.json").is_file():
        health = json.loads((root / "reports" / "collector_health.json").read_text(encoding="utf-8"))
        warnings.append("using reports/collector_health.json instead of health.json")
    elif (root / "collector_health.json").is_file():
        health = json.loads((root / "collector_health.json").read_text(encoding="utf-8"))
        warnings.append("using collector_health.json instead of health.json")

    emitting = health.get("emitting_tokens") or health.get("health", {}).get("emitting_tokens") or []
    if not emitting:
        errors.append("no emitting tokens")

    actual = health.get("duration_sec_actual") or health.get("health", {}).get("duration_sec_actual")
    if actual is None and health.get("health"):
        started = health["health"].get("started_at")
        ended = health["health"].get("ended_at") or health["health"].get("last_message_at")
        if started is not None and ended is not None:
            actual = float(ended) - float(started)
    if actual is None:
        manifest = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = manifest.get("duration_sec")
    if actual is not None and requested_duration_sec is not None and requested_duration_sec >= 30:
        if float(actual) < requested_duration_sec * 0.85:
            errors.append(
                f"early end: actual={float(actual):.1f}s requested={requested_duration_sec:.1f}s"
            )

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored_hashes = manifest.get("artifact_hashes") or {}
        if not stored_hashes:
            errors.append("manifest missing artifact_hashes")
        for rel, expected in stored_hashes.items():
            fp = root / rel
            if fp.is_file():
                got = sha256_file(fp)
                if got != expected:
                    errors.append(f"hash mismatch for {rel}")
    else:
        manifest = {}

    if fp_path.is_file():
        fps = json.loads(fp_path.read_text(encoding="utf-8"))
        if not fps.get("files"):
            errors.append("fingerprints.json missing file entries")

    if not allow_local_abs_paths:
        for fp in (health_path, manifest_path, fp_path):
            if fp.is_file():
                stale = _scan_paths_for_stale(json.loads(fp.read_text(encoding="utf-8")))
                for hit in stale:
                    errors.append(f"stale absolute path in {fp.name}: {hit[:80]}")

    ok = len(errors) == 0
    return {
        "ok": ok,
        "run_dir": str(root),
        "errors": errors,
        "warnings": warnings,
        "raw_files": len(raw_files),
        "normalized_files": len(norm_files),
        "emitting_tokens": len(emitting),
        "manifest": manifest,
    }


def fingerprint_session(run_dir: Path | str) -> dict[str, Any]:
    """Hash raw and normalized session files for integrity tracking."""
    root = Path(run_dir)
    files: dict[str, str] = {}
    for sub in ("raw", "normalized"):
        sub_path = root / sub
        if not sub_path.exists():
            continue
        for fp in sorted(sub_path.rglob("*")):
            if fp.is_file():
                rel = str(fp.relative_to(root))
                files[rel] = sha256_file(fp)
    payload = {
        "session_id": root.name,
        "schema_version": SESSION_SCHEMA_VERSION,
        "files": files,
        "aggregate_hash": hashlib.sha256(
            "".join(f"{k}:{v}" for k, v in sorted(files.items())).encode()
        ).hexdigest(),
    }
    out = root / "fingerprints.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
