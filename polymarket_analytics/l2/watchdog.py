"""Watchdog helpers for scheduled L2 collection health checks."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def watchdog_check(
    *,
    runs: list[dict[str, Any]],
    now: datetime | None = None,
    stale_buffer_minutes: int = 90,
    healthy_window_hours: float = 7.0,
    max_redispatch_per_12h: int = 2,
    redispatch_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Evaluate recent l2-collect workflow runs and recommend recovery actions.

    ``runs`` entries should include: id, status, created_at (ISO), duration_seconds (optional).
    """
    now = now or datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(hours=healthy_window_hours)
    issues: list[str] = []
    latest_success: dict[str, Any] | None = None
    in_progress_healthy = False

    for run in sorted(runs, key=lambda r: r.get("created_at", ""), reverse=True):
        status = run.get("status")
        created_raw = run.get("created_at")
        if not created_raw:
            continue
        created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        dur = float(run.get("duration_seconds") or 18000)
        stale_after = created + timedelta(seconds=dur + stale_buffer_minutes * 60)

        if status == "completed" and run.get("conclusion") == "success":
            if latest_success is None:
                latest_success = run
            continue

        if status == "in_progress" or status == "queued":
            if now <= stale_after:
                in_progress_healthy = True
            else:
                issues.append(f"stale in-progress run {run.get('id')} started {created_raw}")
            continue

        conclusion = run.get("conclusion")
        if status in {"failure", "cancelled", "timed_out"} or (
            status == "completed" and conclusion not in {None, "success"}
        ):
            issues.append(f"run {run.get('id')} status={status} conclusion={conclusion}")

    no_recent_success = latest_success is None or (
        datetime.fromisoformat(latest_success["created_at"].replace("Z", "+00:00")) < stale_cutoff
    )

    unhealthy = no_recent_success and not in_progress_healthy
    state = redispatch_state or {}
    window_start = state.get("window_start")
    count = int(state.get("count") or 0)
    if window_start:
        ws = datetime.fromisoformat(window_start.replace("Z", "+00:00"))
        if now - ws > timedelta(hours=12):
            count = 0
            window_start = None

    should_dispatch = unhealthy and count < max_redispatch_per_12h
    return {
        "ok": not unhealthy,
        "healthy": not unhealthy,
        "unhealthy": unhealthy,
        "issues": issues,
        "latest_success_run_id": latest_success.get("id") if latest_success else None,
        "in_progress_healthy": in_progress_healthy,
        "should_dispatch_replacement": should_dispatch,
        "redispatch_count": count,
        "redispatch_limit": max_redispatch_per_12h,
        "checked_at": now.isoformat(),
    }


def load_redispatch_state(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"count": 0, "window_start": None}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"count": 0, "window_start": None}


def save_redispatch_state(path: Path | str, state: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(state, indent=2), encoding="utf-8")
