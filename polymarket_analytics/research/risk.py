"""Risk and lifecycle control helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RiskLimits:
    max_position_pct: float = 0.05
    max_open_positions: int = 25
    max_gross_exposure_pct: float = 0.50
    max_drawdown_halt_pct: float = 20.0
    per_event_cap_pct: float = 0.10
    cooldown_sec: float = 60.0
    min_time_to_resolution_hours: float | None = None
    max_time_to_resolution_hours: float | None = None


@dataclass
class RiskState:
    equity: float
    cash: float
    peak_equity: float
    n_open: int
    gross_exposure: float
    per_event_exposure: dict[str, float]
    last_entry_ts: float = 0.0

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return 100.0 * (self.peak_equity - self.equity) / self.peak_equity


def check_entry_allowed(
    state: RiskState,
    limits: RiskLimits,
    *,
    notional: float,
    event_id: str,
    now_ts: float,
    ttr_hours: float | None = None,
) -> tuple[bool, str]:
    """Lifecycle/risk gate before opening a position."""
    if state.drawdown_pct >= limits.max_drawdown_halt_pct:
        return False, "max_drawdown_halt"
    if state.n_open >= limits.max_open_positions:
        return False, "max_open_positions"
    if now_ts - state.last_entry_ts < limits.cooldown_sec:
        return False, "cooldown"
    if state.equity > 0 and notional / state.equity > limits.max_position_pct:
        return False, "max_position_pct"
    if state.equity > 0 and (state.gross_exposure + notional) / state.equity > limits.max_gross_exposure_pct:
        return False, "max_gross_exposure"
    ev = state.per_event_exposure.get(event_id, 0.0) + notional
    if state.equity > 0 and ev / state.equity > limits.per_event_cap_pct:
        return False, "per_event_cap"
    if ttr_hours is not None:
        if limits.min_time_to_resolution_hours is not None and ttr_hours < limits.min_time_to_resolution_hours:
            return False, "ttr_too_low"
        if limits.max_time_to_resolution_hours is not None and ttr_hours > limits.max_time_to_resolution_hours:
            return False, "ttr_too_high"
    return True, "ok"


def risk_limits_from_mapping(raw: dict[str, Any] | None) -> RiskLimits:
    if not raw:
        return RiskLimits()
    return RiskLimits(
        max_position_pct=float(raw.get("max_position_pct", 0.05)),
        max_open_positions=int(raw.get("max_open_positions", 25)),
        max_gross_exposure_pct=float(raw.get("max_gross_exposure_pct", 0.50)),
        max_drawdown_halt_pct=float(raw.get("max_drawdown_halt_pct", 20.0)),
        per_event_cap_pct=float(raw.get("per_event_cap_pct", 0.10)),
        cooldown_sec=float(raw.get("cooldown_sec", 60.0)),
        min_time_to_resolution_hours=raw.get("min_time_to_resolution_hours"),
        max_time_to_resolution_hours=raw.get("max_time_to_resolution_hours"),
    )
