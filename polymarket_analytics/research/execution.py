"""Execution simulation foundations: latency, book walk, partial fills, markout.

Honest gaps: full LOB event reconstruction / queue-ahead without L3 data is blocked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

from polymarket_analytics.research.fees import FeeModel, compute_fill_fee

LiquidityRole = Literal["maker", "taker", "hybrid"]


@dataclass(frozen=True)
class BookLevel:
    price: float
    size: float


@dataclass(frozen=True)
class ExecutionConfig:
    latency_ms: float = 50.0
    partial_fill: bool = True
    touch_does_not_fill: bool = True  # resting at touch ≠ automatic fill
    max_levels: int = 5
    role: LiquidityRole = "taker"
    markout_horizon_s: float = 60.0
    fee_category: str = "crypto"
    fee_model_version: str = ""  # filled from FeeModel if empty


@dataclass(frozen=True)
class FillResult:
    filled_size: float
    avg_price: float
    fee: float
    levels_consumed: int
    residual_size: float
    role: str
    markout: float | None = None
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def walk_book(
    side: Literal["buy", "sell"],
    size: float,
    levels: Sequence[BookLevel],
    *,
    partial_fill: bool = True,
) -> tuple[float, float, int, float]:
    """
    Walk ask levels for buys / bid levels for sells.

    Returns (filled_size, notional, levels_consumed, residual_size).
    """
    remaining = max(float(size), 0.0)
    filled = 0.0
    notional = 0.0
    consumed = 0
    for lvl in levels:
        if remaining <= 0:
            break
        take = min(remaining, max(float(lvl.size), 0.0))
        if take <= 0:
            continue
        filled += take
        notional += take * float(lvl.price)
        remaining -= take
        consumed += 1
        if not partial_fill and remaining > 0:
            # All-or-nothing failure
            return 0.0, 0.0, 0, float(size)
    avg = notional / filled if filled > 0 else 0.0
    return filled, avg, consumed, remaining


def simulate_aggressive_fill(
    side: Literal["buy", "sell"],
    size: float,
    levels: Sequence[BookLevel],
    *,
    cfg: ExecutionConfig | None = None,
    mid_after: float | None = None,
    as_of: str | None = None,
) -> FillResult:
    """Taker-style book walk + per-fill fee + optional adverse markout."""
    cfg = cfg or ExecutionConfig()
    filled, avg, consumed, residual = walk_book(
        side, size, levels, partial_fill=cfg.partial_fill
    )
    role = "taker" if cfg.role != "maker" else "maker"
    fee_info = compute_fill_fee(
        filled,
        avg if avg > 0 else (levels[0].price if levels else 0.5),
        role=role,  # type: ignore[arg-type]
        category=cfg.fee_category,
        as_of=as_of,
        model=FeeModel(),
    )
    markout = None
    if mid_after is not None and filled > 0 and avg > 0:
        # Adverse markout: buy → mid drop hurts; sell → mid rise hurts
        if side == "buy":
            markout = avg - float(mid_after)
        else:
            markout = float(mid_after) - avg
    return FillResult(
        filled_size=filled,
        avg_price=avg,
        fee=float(fee_info["fee"]),
        levels_consumed=consumed,
        residual_size=residual,
        role=role,
        markout=markout,
        meta={
            "latency_ms": cfg.latency_ms,
            "fee_model_version": fee_info["fee_model_version"],
            "touch_does_not_fill": cfg.touch_does_not_fill,
            "markout_horizon_s": cfg.markout_horizon_s,
        },
    )


def queue_ahead_fill_probability(
    queue_ahead: float,
    trade_through: float,
    *,
    touch_does_not_fill: bool = True,
) -> float:
    """
    Crude maker fill probability given queue ahead and volume traded through.

    If touch_does_not_fill, require trade_through > queue_ahead (strict).
    """
    if trade_through <= 0 or queue_ahead < 0:
        return 0.0
    if touch_does_not_fill and trade_through <= queue_ahead:
        return 0.0
    # Linear depletion heuristic
    return min(1.0, max(0.0, (trade_through - queue_ahead) / max(queue_ahead, 1e-9)))


def order_tp_sl_tick(
    *,
    entry: float,
    mark: float,
    take_profit_pct: float | None,
    stop_loss_pct: float | None,
    prefer: Literal["stop_first", "tp_first"] = "stop_first",
) -> Literal["none", "take_profit", "stop_loss"]:
    """
    Deterministic TP/SL ordering on a single tick (no path ambiguity).

    When both hit on same mark, ``prefer`` decides — default stop_first (conservative).
    """
    if entry <= 0:
        return "none"
    ret = (mark - entry) / entry
    hit_tp = take_profit_pct is not None and ret >= take_profit_pct
    hit_sl = stop_loss_pct is not None and ret <= -stop_loss_pct
    if hit_tp and hit_sl:
        return "stop_loss" if prefer == "stop_first" else "take_profit"
    if hit_sl:
        return "stop_loss"
    if hit_tp:
        return "take_profit"
    return "none"


EXECUTION_GAPS: tuple[str, ...] = (
    "No L3 order-id queue reconstruction from available trade/book snapshots.",
    "Latency is a fixed delay parameter — not measured exchange RTT.",
    "Hybrid maker/taker path switching requires live cancel/replace semantics.",
    "Adverse markout needs future mid; only available when mid_after supplied.",
)
