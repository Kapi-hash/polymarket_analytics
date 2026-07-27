"""Conservative taker-only outcome execution without order books."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import polars as pl

from polymarket_analytics.research.historical_fees import compute_historical_fill_fee

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class OutcomeExecConfig:
    """Taker proxy config — no maker / queue claims."""

    latency_s: float = 1.0
    slip_bps: float = 50.0
    entry_timeout_s: float = 300.0
    shares: float = 1.0
    fee_category: str | None = None
    allow_maker: bool = False  # must stay False for honest labeling


@dataclass(frozen=True)
class OutcomeFill:
    """Result of next-print adverse fill simulation."""

    filled: bool
    fill_price: float | None
    signal_price: float
    print_price: float | None
    fee: float
    side: Side
    token_id: str
    signal_at: datetime
    fill_at: datetime | None
    reject_reason: str | None = None
    execution_label: str = "next_print_adverse_taker"
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_side(raw: str | None, default: Side = "buy") -> Side:
    if raw is None:
        return default
    text = str(raw).strip().upper()
    if text in {"BUY", "B", "BID"}:
        return "buy"
    if text in {"SELL", "S", "ASK"}:
        return "sell"
    return default


def _clip_price(px: float) -> float:
    return min(max(float(px), 0.01), 0.99)


def adverse_fill_price(
    signal_px: float,
    print_px: float,
    side: Side,
    slip_bps: float,
) -> float:
    """
    Adverse taker fill: buy at max(signal, print) + slip; sell at min - slip.
    """
    slip = max(float(slip_bps), 0.0) / 10_000.0
    if side == "buy":
        base = max(float(signal_px), float(print_px))
        return _clip_price(base * (1.0 + slip))
    base = min(float(signal_px), float(print_px))
    return _clip_price(base * (1.0 - slip))


def find_next_eligible_print(
    prints: pl.DataFrame,
    *,
    token_id: str,
    side: Side,
    after: datetime,
    timeout_s: float,
) -> pl.DataFrame:
    """
    Next trade print on ``token_id`` after ``after`` within ``timeout_s``.

    ``prints`` must have token_id, traded_at/timestamp, price, optional side.
    """
    time_col = "traded_at" if "traded_at" in prints.columns else "timestamp"
    token_col = "token_id" if "token_id" in prints.columns else "asset_id"
    deadline = _to_utc(after) + timedelta(seconds=float(timeout_s))

    filt = prints.filter(
        (pl.col(token_col) == token_id)
        & (pl.col(time_col) > pl.lit(_to_utc(after)))
        & (pl.col(time_col) <= pl.lit(deadline))
    ).sort(time_col)

    if side_col := next((c for c in ("side", "maker_direction", "taker_direction") if c in filt.columns), None):
        side_upper = side.upper()
        filt = filt.filter(pl.col(side_col).cast(pl.Utf8).str.to_uppercase().str.contains(side_upper))

    return filt.head(1)


def simulate_outcome_fill(
    prints: pl.DataFrame,
    *,
    token_id: str,
    signal_at: datetime,
    signal_price: float,
    side: Side = "buy",
    cfg: OutcomeExecConfig | None = None,
) -> OutcomeFill:
    """
    Conservative taker proxy: next eligible print after signal + latency.

    Rejects if no print within entry_timeout_s. Charges historical fees.
    """
    cfg = cfg or OutcomeExecConfig()
    if cfg.allow_maker:
        raise ValueError("Maker fills disallowed — use taker proxy only")

    signal_at = _to_utc(signal_at)
    eligible_after = signal_at + timedelta(seconds=float(cfg.latency_s))
    next_print = find_next_eligible_print(
        prints,
        token_id=token_id,
        side=side,
        after=eligible_after,
        timeout_s=cfg.entry_timeout_s,
    )

    time_col = "traded_at" if "traded_at" in prints.columns else "timestamp"

    if next_print.is_empty():
        return OutcomeFill(
            filled=False,
            fill_price=None,
            signal_price=float(signal_price),
            print_price=None,
            fee=0.0,
            side=side,
            token_id=token_id,
            signal_at=signal_at,
            fill_at=None,
            reject_reason="no_print_within_timeout",
            meta={"latency_s": cfg.latency_s, "timeout_s": cfg.entry_timeout_s},
        )

    row = next_print.row(0, named=True)
    print_px = float(row["price"])
    fill_px = adverse_fill_price(signal_price, print_px, side, cfg.slip_bps)
    fill_at = row[time_col]
    if isinstance(fill_at, datetime):
        fill_at = _to_utc(fill_at)

    fee_info = compute_historical_fill_fee(
        cfg.shares,
        fill_px,
        as_of=fill_at,
        role="taker",
        category=cfg.fee_category,
    )

    return OutcomeFill(
        filled=True,
        fill_price=fill_px,
        signal_price=float(signal_price),
        print_price=print_px,
        fee=float(fee_info["fee"]),
        side=side,
        token_id=token_id,
        signal_at=signal_at,
        fill_at=fill_at,
        reject_reason=None,
        meta={
            "latency_s": cfg.latency_s,
            "slip_bps": cfg.slip_bps,
            "fee_regime": fee_info.get("fee_regime"),
            "fee_confidence": fee_info.get("fee_confidence"),
            "fee_model_version": fee_info.get("fee_model_version"),
        },
    )
