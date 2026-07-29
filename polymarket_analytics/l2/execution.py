"""L2 execution simulation: taker book walk, maker fill bounds, latency scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

from polymarket_analytics.research.fees import FeeModel, compute_fill_fee

LATENCY_SCENARIOS_MS: tuple[int, ...] = (50, 100, 250, 500, 1000)
DEFAULT_TICK: float = 0.01
DEFAULT_MIN_SIZE: float = 1.0

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class BookLevel:
    price: float
    size: float


@dataclass
class TakerExecutionConfig:
    latency_ms: float = 50.0
    partial_fill: bool = True
    tick_size: float = DEFAULT_TICK
    min_size: float = DEFAULT_MIN_SIZE
    max_staleness_ms: float = 500.0
    fee_category: str = "crypto"
    as_of: str | None = None


@dataclass
class MakerExecutionConfig:
    latency_ms: float = 100.0
    queue_ahead: float = 0.0
    touch_does_not_fill: bool = True
    fee_category: str = "crypto"
    as_of: str | None = None


@dataclass
class TakerFillResult:
    filled_size: float
    avg_price: float
    fee: float
    levels_consumed: int
    residual_size: float
    realized_spread: float | None
    adverse_selection: float | None
    rejected: bool = False
    reject_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MakerFillBounds:
    lower_prob: float
    base_prob: float
    upper_prob: float
    queue_ahead: float
    trade_through: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def round_to_tick(price: float, *, tick: float = DEFAULT_TICK) -> float:
    if tick <= 0:
        return float(price)
    return round(float(price) / tick) * tick


def round_size(size: float, *, min_size: float = DEFAULT_MIN_SIZE) -> float:
    if min_size <= 0:
        return float(size)
    steps = round(float(size) / min_size)
    return max(0.0, steps * min_size)


def walk_book_taker(
    side: Side,
    size: float,
    levels: Sequence[BookLevel],
    *,
    partial_fill: bool = True,
    tick_size: float = DEFAULT_TICK,
    min_size: float = DEFAULT_MIN_SIZE,
) -> tuple[float, float, int, float]:
    """Walk book for taker; returns filled, avg_price, levels_consumed, residual."""
    remaining = round_size(size, min_size=min_size)
    filled = 0.0
    notional = 0.0
    consumed = 0
    ordered = sorted(levels, key=lambda l: l.price, reverse=(side == "sell"))
    for lvl in ordered:
        if remaining <= 0:
            break
        take = min(remaining, round_size(lvl.size, min_size=min_size))
        if take <= 0:
            continue
        px = round_to_tick(lvl.price, tick=tick_size)
        filled += take
        notional += take * px
        remaining -= take
        consumed += 1
        if not partial_fill and remaining > 0:
            return 0.0, 0.0, 0, float(size)
    avg = notional / filled if filled > 0 else 0.0
    return filled, avg, consumed, remaining


def simulate_taker_execution(
    side: Side,
    size: float,
    *,
    bids: Sequence[BookLevel],
    asks: Sequence[BookLevel],
    mid_at_entry: float | None,
    mid_after: float | None = None,
    book_uncertain: bool = False,
    book_age_ms: float = 0.0,
    cfg: TakerExecutionConfig | None = None,
) -> TakerFillResult:
    """Simulate taker execution with stale/uncertain rejection and fee hook."""
    cfg = cfg or TakerExecutionConfig()
    if book_uncertain:
        return TakerFillResult(
            0, 0, 0, 0, 0, None, None, True, "uncertain_book",
            {"latency_ms": cfg.latency_ms},
        )
    if book_age_ms > cfg.max_staleness_ms + cfg.latency_ms:
        return TakerFillResult(
            0, 0, 0, 0, 0, None, None, True, "stale_book",
            {"latency_ms": cfg.latency_ms, "book_age_ms": book_age_ms},
        )
    levels = asks if side == "buy" else bids
    filled, avg, consumed, residual = walk_book_taker(
        side, size, levels, partial_fill=cfg.partial_fill,
        tick_size=cfg.tick_size, min_size=cfg.min_size,
    )
    fee_info = compute_fill_fee(
        filled, avg if avg > 0 else 0.5, role="taker",
        category=cfg.fee_category, as_of=cfg.as_of, model=FeeModel(),
    )
    realized_spread = None
    adverse = None
    if mid_at_entry is not None and filled > 0 and avg > 0:
        half_spread = abs(float(avg) - float(mid_at_entry))
        realized_spread = half_spread * 2.0
        if mid_after is not None:
            adverse = (float(avg) - float(mid_after)) if side == "buy" else (float(mid_after) - float(avg))
    return TakerFillResult(
        filled_size=filled,
        avg_price=avg,
        fee=float(fee_info["fee"]),
        levels_consumed=consumed,
        residual_size=residual,
        realized_spread=realized_spread,
        adverse_selection=adverse,
        meta={"latency_ms": cfg.latency_ms, "fee_model_version": fee_info["fee_model_version"]},
    )


def maker_fill_bounds(
    *,
    queue_ahead: float,
    trade_through: float,
    visible_depth: float,
    cfg: MakerExecutionConfig | None = None,
) -> MakerFillBounds:
    """Conservative lower/base/upper maker fill probability bounds (no exact queue claim)."""
    cfg = cfg or MakerExecutionConfig()
    qa = max(float(queue_ahead), 0.0)
    tt = max(float(trade_through), 0.0)
    depth = max(float(visible_depth), 1e-9)

    if cfg.touch_does_not_fill and tt <= qa:
        return MakerFillBounds(0.0, 0.0, 0.0, qa, tt, notes="touch_does_not_fill")

    base = min(1.0, max(0.0, (tt - qa) / depth))
    lower = max(0.0, base * 0.5)
    upper = min(1.0, base * 1.25 + (0.1 if tt > qa else 0.0))
    return MakerFillBounds(lower, base, upper, qa, tt, notes="conservative_bounds")
