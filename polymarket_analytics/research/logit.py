"""Logit-space foundations for probability markets (half-tick clamp)."""

from __future__ import annotations

import math
from typing import Final

import polars as pl

# Polymarket CLOB typical tick; clamp away from {0,1} by half a tick.
DEFAULT_HALF_TICK: Final[float] = 0.005


def clamp_prob(p: float, *, half_tick: float = DEFAULT_HALF_TICK) -> float:
    """Clamp probability into (half_tick, 1 - half_tick)."""
    lo = float(half_tick)
    hi = 1.0 - lo
    if hi <= lo:
        raise ValueError("half_tick must be < 0.5")
    return min(max(float(p), lo), hi)


def logit(p: float, *, half_tick: float = DEFAULT_HALF_TICK) -> float:
    """log(p / (1-p)) with half-tick clamp."""
    x = clamp_prob(p, half_tick=half_tick)
    return math.log(x / (1.0 - x))


def sigmoid_from_logit(z: float) -> float:
    """Inverse logit (logistic)."""
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def logit_edge(
    p_model: float,
    p_market: float,
    *,
    half_tick: float = DEFAULT_HALF_TICK,
) -> float:
    """Logit-space edge: logit(p_model) - logit(p_market)."""
    return logit(p_model, half_tick=half_tick) - logit(p_market, half_tick=half_tick)


def logit_expr(
    price: pl.Expr,
    *,
    half_tick: float = DEFAULT_HALF_TICK,
    alias: str = "logit_price",
) -> pl.Expr:
    """Vectorized logit of a probability column (Polars)."""
    lo = float(half_tick)
    hi = 1.0 - lo
    clamped = price.clip(lower_bound=lo, upper_bound=hi)
    return (clamped / (1.0 - clamped)).log().alias(alias)


def attach_logit_columns(
    df: pl.DataFrame,
    *,
    price_col: str = "price",
    half_tick: float = DEFAULT_HALF_TICK,
) -> pl.DataFrame:
    """Attach logit_price and delta_logit_1h (if price_delta_1h present via lag)."""
    if df.is_empty() or price_col not in df.columns:
        return df
    out = df.with_columns(logit_expr(pl.col(price_col), half_tick=half_tick))
    if "token_id" in out.columns and "traded_at" in out.columns:
        out = out.sort(["token_id", "traded_at"]).with_columns(
            (pl.col("logit_price") - pl.col("logit_price").shift(1).over("token_id")).alias(
                "delta_logit_1bar"
            )
        )
    return out
