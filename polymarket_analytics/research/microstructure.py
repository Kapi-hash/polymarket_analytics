"""Order-book / microstructure helpers (OFI, resilience, event decomposition).

Full L2 reconstruction is blocked without historical depth events.
These APIs accept level frames when available and otherwise return nulls.
"""

from __future__ import annotations

from typing import Sequence

import polars as pl


def compute_ofi_from_levels(
    bid_sizes: Sequence[float],
    ask_sizes: Sequence[float],
    *,
    prev_bid_sizes: Sequence[float] | None = None,
    prev_ask_sizes: Sequence[float] | None = None,
) -> float:
    """
    Cont-style multi-level OFI approximation.

    Without previous book: signed depth imbalance Σ(bid_i - ask_i) / Σ(bid_i + ask_i).
    With previous book: sum of signed size changes at overlapping levels.
    """
    n = min(len(bid_sizes), len(ask_sizes))
    if n == 0:
        return 0.0
    if prev_bid_sizes is None or prev_ask_sizes is None:
        num = sum(float(bid_sizes[i]) - float(ask_sizes[i]) for i in range(n))
        den = sum(float(bid_sizes[i]) + float(ask_sizes[i]) for i in range(n))
        return num / den if den > 0 else 0.0

    pn = min(n, len(prev_bid_sizes), len(prev_ask_sizes))
    ofi = 0.0
    for i in range(pn):
        db = float(bid_sizes[i]) - float(prev_bid_sizes[i])
        da = float(ask_sizes[i]) - float(prev_ask_sizes[i])
        ofi += db - da
    return ofi


def attach_top_of_book_imbalance(df: pl.DataFrame) -> pl.DataFrame:
    """Attach book_imbalance = bid_depth / ask_depth when columns exist."""
    if "bid_depth" not in df.columns or "ask_depth" not in df.columns:
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias("book_imbalance"))
    return df.with_columns(
        pl.when(pl.col("ask_depth").is_not_null() & (pl.col("ask_depth") > 0))
        .then(pl.col("bid_depth") / pl.col("ask_depth"))
        .otherwise(None)
        .alias("book_imbalance")
    )


def book_resilience_proxy(
    depth_before: float,
    depth_after: float,
    *,
    eps: float = 1e-9,
) -> float:
    """Fraction of depth restored after a hit (1 = full resilience)."""
    if depth_before <= eps:
        return 1.0
    return max(0.0, min(1.0, float(depth_after) / float(depth_before)))


def decompose_trade_aggression(
    price: float,
    best_bid: float | None,
    best_ask: float | None,
) -> str:
    """Classify trade as buy_aggressive / sell_aggressive / mid / unknown."""
    if best_bid is None or best_ask is None:
        return "unknown"
    if price >= best_ask:
        return "buy_aggressive"
    if price <= best_bid:
        return "sell_aggressive"
    return "mid"


def attach_trade_aggression(df: pl.DataFrame) -> pl.DataFrame:
    """Vectorized aggression labels when best_bid/best_ask present."""
    need = {"price", "best_bid", "best_ask"}
    if not need.issubset(df.columns):
        return df.with_columns(pl.lit("unknown").alias("trade_aggression"))
    return df.with_columns(
        pl.when(pl.col("price") >= pl.col("best_ask"))
        .then(pl.lit("buy_aggressive"))
        .when(pl.col("price") <= pl.col("best_bid"))
        .then(pl.lit("sell_aggressive"))
        .otherwise(pl.lit("mid"))
        .alias("trade_aggression")
    )
