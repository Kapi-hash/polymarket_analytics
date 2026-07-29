"""L2 research readiness gates — fail closed on thin samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

MIN_CALENDAR_DAYS: Final[int] = 7
MIN_EMITTING_TOKENS: Final[int] = 50
MIN_INDEPENDENT_EVENTS: Final[int] = 20
THIN_SAMPLE_HOURS: Final[float] = 5.0
DIAGNOSTIC_ONLY_NOTE: Final[str] = (
    "Samples under 7 calendar days are diagnostics-only and non-definitive."
)


def evaluate_readiness(coverage_json: Path | str) -> dict[str, Any]:
    """Load daily coverage JSON and evaluate readiness (non-definitive gate)."""
    path = Path(coverage_json)
    if not path.exists():
        return {"ok": False, "ready": False, "error": f"missing coverage json: {path}"}
    try:
        cov = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"ok": False, "ready": False, "error": f"invalid json: {exc}"}

    # Fail closed: do not invent dayparts, buckets, fees, or purged splits.
    # Coverage JSON must supply explicit readiness fields (or nested "readiness").
    nested = cov.get("readiness") if isinstance(cov.get("readiness"), dict) else {}
    adapted = {
        "calendar_days": nested.get("calendar_days", cov.get("calendar_days", 0)),
        "emitting_tokens": nested.get(
            "emitting_tokens",
            cov.get("n_emitting_tokens_union") or cov.get("emitting_tokens") or 0,
        ),
        "independent_events": nested.get(
            "independent_events",
            cov.get("independent_events") or cov.get("n_events") or 0,
        ),
        "utc_dayparts": nested.get("utc_dayparts", cov.get("utc_dayparts") or []),
        "probability_buckets": nested.get(
            "probability_buckets", cov.get("probability_buckets") or []
        ),
        "ttr_regimes": nested.get("ttr_regimes", cov.get("ttr_regimes") or []),
        "pit_metadata_ok": nested.get(
            "pit_metadata_ok", cov.get("pit_metadata_ok", False)
        ),
        "fees_valid": nested.get("fees_valid", cov.get("fees_valid", False)),
        "splits": nested.get("splits", cov.get("splits") or {}),
        "gap_rate": nested.get("gap_rate", cov.get("gap_rate")),
        "collection_hours": nested.get(
            "collection_hours",
            cov.get("collection_hours")
            or float(cov.get("covered_hours") or 0)
            or float(cov.get("n_sessions") or 0) * 5.0,
        ),
    }
    result = evaluate_l2_readiness(adapted)
    ready = result.get("decision") == "RESEARCH_READY"
    return {
        "ok": True,
        "ready": ready,
        "decision": result.get("decision"),
        "reasons": result.get("reasons"),
        "warnings": result.get("warnings"),
        "coverage_path": str(path),
        "non_definitive": result.get("non_definitive", True),
        "note": "Readiness permits staged diagnostics only; not authorization for broad outcome sweeps.",
    }


def evaluate_l2_readiness(coverage: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate whether L2 microstructure research is authorized.

    Returns decision RESEARCH_READY or NOT_READY with explicit reasons.
    """
    reasons: list[str] = []
    warnings: list[str] = []

    calendar_days = float(coverage.get("calendar_days") or 0)
    if calendar_days < MIN_CALENDAR_DAYS:
        reasons.append(f"calendar_days={calendar_days:.1f} < {MIN_CALENDAR_DAYS}")

    dayparts = coverage.get("utc_dayparts") or coverage.get("dayparts") or []
    if len(dayparts) < 2:
        reasons.append("insufficient UTC daypart coverage")

    emitting = int(coverage.get("emitting_tokens") or coverage.get("n_emitting_tokens") or 0)
    if emitting < MIN_EMITTING_TOKENS:
        reasons.append(f"emitting_tokens={emitting} < {MIN_EMITTING_TOKENS}")

    events = int(coverage.get("independent_events") or coverage.get("n_events") or 0)
    if events < MIN_INDEPENDENT_EVENTS:
        reasons.append(f"independent_events={events} < {MIN_INDEPENDENT_EVENTS}")

    prob_buckets = coverage.get("probability_buckets") or []
    ttr_regimes = coverage.get("ttr_regimes") or []
    if len(prob_buckets) < 2:
        reasons.append("insufficient probability bucket diversity")
    if len(ttr_regimes) < 2:
        reasons.append("insufficient TTR regime diversity")

    gap_rate = coverage.get("gap_rate")
    if gap_rate is not None and float(gap_rate) > float(coverage.get("max_gap_rate", 0.05)):
        reasons.append(f"gap_rate={float(gap_rate):.4f} exceeds threshold")

    if not coverage.get("pit_metadata_ok", False):
        reasons.append("PIT metadata incomplete")

    if not coverage.get("fees_valid", False):
        reasons.append("fee metadata invalid or missing")

    splits = coverage.get("splits") or {}
    for name in ("train", "val", "locked_test"):
        if not splits.get(name):
            reasons.append(f"purged split '{name}' is empty")

    hours = coverage.get("collection_hours")
    if hours is not None and float(hours) <= THIN_SAMPLE_HOURS:
        warnings.append(f"{float(hours):.1f}h sample marked diagnostics-only (non-definitive)")

    if calendar_days < MIN_CALENDAR_DAYS:
        warnings.append(DIAGNOSTIC_ONLY_NOTE)

    decision = "RESEARCH_READY" if not reasons else "NOT_READY"
    return {
        "decision": decision,
        "research_authorized": decision == "RESEARCH_READY",
        "paper_authorized": False,
        "reasons": reasons,
        "warnings": warnings,
        "diagnostic_only": calendar_days < MIN_CALENDAR_DAYS,
        "non_definitive": calendar_days < MIN_CALENDAR_DAYS,
        "minimums": {
            "calendar_days": MIN_CALENDAR_DAYS,
            "emitting_tokens": MIN_EMITTING_TOKENS,
            "independent_events": MIN_INDEPENDENT_EVENTS,
        },
    }


def assert_not_authorized_from_thin_sample(hours: float) -> None:
    """Raise if attempting definitive sweep on insufficient collection window."""
    days = float(hours) / 24.0
    if days < MIN_CALENDAR_DAYS:
        raise PermissionError(
            f"Definitive L2 sweep blocked: {hours:.1f}h ({days:.2f} days) < {MIN_CALENDAR_DAYS} days. "
            f"Use diagnostics-only mode."
        )
