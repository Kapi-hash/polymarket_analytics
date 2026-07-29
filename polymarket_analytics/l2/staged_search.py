"""Staged L2 search scaffold — refuses definitive stages unless readiness passes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from polymarket_analytics.l2.readiness import assert_not_authorized_from_thin_sample, evaluate_l2_readiness

StageStatus = Literal["pending", "running", "completed", "blocked", "diagnostic_only"]


@dataclass
class StagedSearchConfig:
    allow_diagnostics_on_thin_data: bool = True
    definitive_stages: tuple[int, ...] = (3, 4, 5)
    diagnostic_stages: tuple[int, ...] = (1, 2)


@dataclass
class StageResult:
    stage: int
    status: StageStatus
    research_authorized: bool = False
    paper_authorized: bool = False
    non_definitive: bool = False
    message: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_staged_search(
    *,
    stage: int,
    coverage: dict[str, Any],
    cfg: StagedSearchConfig | None = None,
) -> StageResult:
    """Run or gate a staged search stage."""
    cfg = cfg or StagedSearchConfig()
    readiness = evaluate_l2_readiness(coverage)
    hours = float(coverage.get("collection_hours") or 0)

    if stage in cfg.definitive_stages:
        if readiness["decision"] != "RESEARCH_READY":
            return StageResult(
                stage=stage,
                status="blocked",
                message=f"Stage {stage} blocked: readiness={readiness['decision']}",
                meta={"reasons": readiness["reasons"]},
            )
        try:
            assert_not_authorized_from_thin_sample(hours)
        except PermissionError as exc:
            return StageResult(stage=stage, status="blocked", message=str(exc))
        return StageResult(
            stage=stage,
            status="completed",
            research_authorized=True,
            message=f"Stage {stage} authorized",
        )

    if stage in cfg.diagnostic_stages:
        if not cfg.allow_diagnostics_on_thin_data and readiness["decision"] != "RESEARCH_READY":
            return StageResult(
                stage=stage,
                status="blocked",
                message="Diagnostic stages disabled on thin data",
            )
        return StageResult(
            stage=stage,
            status="diagnostic_only",
            non_definitive=True,
            message=f"Stage {stage} diagnostic-only (labels permitted)",
            meta={"warnings": readiness["warnings"]},
        )

    return StageResult(stage=stage, status="blocked", message=f"Unknown stage {stage}")
