"""Configurable swing lifecycle with gap-aware exit rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from polymarket_analytics.research.logit import logit

ExitReason = Literal[
    "take_profit", "stop_loss", "trailing", "break_even", "time_stop",
    "volume_stop", "activity_stop", "liquidity_deterioration", "spread_blowout",
    "imbalance_reversal", "toxic_flow", "ttr_forced", "no_trade_zone", "gap_forced", "open",
]


@dataclass
class LifecycleConfig:
    take_profit_pct: float | None = 0.05
    stop_loss_pct: float | None = 0.03
    trailing_pct: float | None = None
    break_even_trigger_pct: float | None = None
    partial_tp_pct: float | None = None
    partial_tp_fraction: float = 0.5
    max_hold_sec: float | None = None
    max_volume_sec: float | None = None
    spread_blowout_mult: float = 3.0
    imbalance_reversal_threshold: float = 0.5
    ttr_forced_exit_hours: float | None = 1.0
    allow_gap_hold: bool = False
    atr_scaled: bool = False
    logit_vol_scaled: bool = False


@dataclass
class LifecycleState:
    entry_price: float
    entry_time: float
    side: Literal["long", "short"] = "long"
    size: float = 1.0
    remaining_size: float = 1.0
    mfe: float = 0.0
    mae: float = 0.0
    high_water: float | None = None
    break_even_active: bool = False
    partial_taken: bool = False
    exit_reason: ExitReason = "open"
    exit_price: float | None = None
    exit_time: float | None = None
    in_gap: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _return_pct(entry: float, mark: float, side: str) -> float:
    if entry <= 0:
        return 0.0
    raw = (mark - entry) / entry
    return raw if side == "long" else -raw


def evaluate_lifecycle_tick(
    state: LifecycleState,
    *,
    mark: float,
    now: float,
    spread: float | None = None,
    entry_spread: float | None = None,
    imbalance: float | None = None,
    entry_imbalance: float | None = None,
    ttr_hours: float | None = None,
    in_gap: bool = False,
    cfg: LifecycleConfig | None = None,
    logit_vol: float | None = None,
) -> LifecycleState:
    """Advance lifecycle one tick; refuse holding across gaps unless allowed."""
    cfg = cfg or LifecycleConfig()
    state.in_gap = in_gap
    if in_gap and not cfg.allow_gap_hold:
        state.exit_reason = "gap_forced"
        state.exit_price = mark
        state.exit_time = now
        return state

    ret = _return_pct(state.entry_price, mark, state.side)
    state.mfe = max(state.mfe, ret)
    state.mae = min(state.mae, ret)

    if state.high_water is None:
        state.high_water = mark
    if state.side == "long":
        state.high_water = max(state.high_water, mark)
    else:
        state.high_water = min(state.high_water, mark)

    tp = cfg.take_profit_pct
    sl = cfg.stop_loss_pct
    if cfg.logit_vol_scaled and logit_vol is not None and logit_vol > 0:
        tp = (tp or 0.05) * logit_vol
        sl = (sl or 0.03) * logit_vol

    if cfg.break_even_trigger_pct is not None and ret >= cfg.break_even_trigger_pct:
        state.break_even_active = True
        sl = 0.0

    if tp is not None and ret >= tp:
        if cfg.partial_tp_pct is not None and not state.partial_taken and ret >= cfg.partial_tp_pct:
            state.remaining_size *= (1.0 - cfg.partial_tp_fraction)
            state.partial_taken = True
            state.meta["partial_tp_at"] = now
        else:
            state.exit_reason = "take_profit"
            state.exit_price = mark
            state.exit_time = now
            return state

    eff_sl = sl
    if cfg.trailing_pct is not None and state.high_water is not None:
        trail_ret = _return_pct(state.entry_price, state.high_water, state.side)
        if trail_ret - ret >= cfg.trailing_pct:
            state.exit_reason = "trailing"
            state.exit_price = mark
            state.exit_time = now
            return state

    if eff_sl is not None and ret <= -abs(eff_sl):
        state.exit_reason = "stop_loss"
        state.exit_price = mark
        state.exit_time = now
        return state

    if cfg.max_hold_sec is not None and (now - state.entry_time) >= cfg.max_hold_sec:
        state.exit_reason = "time_stop"
        state.exit_price = mark
        state.exit_time = now
        return state

    if spread is not None and entry_spread is not None and spread > entry_spread * cfg.spread_blowout_mult:
        state.exit_reason = "spread_blowout"
        state.exit_price = mark
        state.exit_time = now
        return state

    if imbalance is not None and entry_imbalance is not None:
        if state.side == "long" and imbalance < entry_imbalance - cfg.imbalance_reversal_threshold:
            state.exit_reason = "imbalance_reversal"
            state.exit_price = mark
            state.exit_time = now
            return state

    if ttr_hours is not None and cfg.ttr_forced_exit_hours is not None:
        if ttr_hours <= cfg.ttr_forced_exit_hours:
            state.exit_reason = "ttr_forced"
            state.exit_price = mark
            state.exit_time = now
            return state

    return state


def logit_stop_distance(prob: float, stop_logit: float) -> float:
    """Distance in probability space implied by logit stop."""
    return abs(logit(prob) - stop_logit)
