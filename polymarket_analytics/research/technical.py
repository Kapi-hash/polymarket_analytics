"""Logit-space technical indicators (clock/trade bars; Polars-first)."""

from __future__ import annotations

import math
from typing import Sequence

import polars as pl

from polymarket_analytics.research.logit import DEFAULT_HALF_TICK, clamp_prob, logit


def arcsine_sqrt(p: float, *, half_tick: float = DEFAULT_HALF_TICK) -> float:
    """y = 2 arcsin(sqrt(p)) with half-tick clamp."""
    x = clamp_prob(p, half_tick=half_tick)
    return 2.0 * math.asin(math.sqrt(x))


def arcsine_expr(
    price: pl.Expr,
    *,
    half_tick: float = DEFAULT_HALF_TICK,
    alias: str = "arcsine_price",
) -> pl.Expr:
    lo = float(half_tick)
    hi = 1.0 - lo
    clamped = price.clip(lower_bound=lo, upper_bound=hi)
    return (2.0 * clamped.sqrt().arcsin()).alias(alias)


def attach_arcsine(df: pl.DataFrame, *, price_col: str = "price") -> pl.DataFrame:
    if df.is_empty() or price_col not in df.columns:
        return df
    return df.with_columns(arcsine_expr(pl.col(price_col)))


def _rsi_from_deltas(deltas: Sequence[float], period: int) -> float | None:
    if len(deltas) < period:
        return None
    window = deltas[-period:]
    gains = [d for d in window if d > 0]
    losses = [-d for d in window if d < 0]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss <= 1e-12:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def logit_rsi(
    prices: Sequence[float],
    *,
    period: int = 14,
    half_tick: float = DEFAULT_HALF_TICK,
) -> float | None:
    """RSI on logit(price) differences."""
    if len(prices) < period + 1:
        return None
    zs = [logit(p, half_tick=half_tick) for p in prices]
    deltas = [zs[i] - zs[i - 1] for i in range(1, len(zs))]
    return _rsi_from_deltas(deltas, period)


def ema_series(values: Sequence[float], span: int) -> list[float | None]:
    if span <= 0:
        raise ValueError("span must be positive")
    alpha = 2.0 / (span + 1.0)
    out: list[float | None] = []
    ema: float | None = None
    for v in values:
        if ema is None:
            ema = float(v)
        else:
            ema = alpha * float(v) + (1.0 - alpha) * ema
        out.append(ema)
    return out


def logit_ema_cross(
    prices: Sequence[float],
    *,
    fast: int = 5,
    slow: int = 20,
    half_tick: float = DEFAULT_HALF_TICK,
) -> dict[str, float | bool | None]:
    """EMA cross on logit prices; bullish when fast > slow."""
    if len(prices) < slow + 1:
        return {"ema_fast": None, "ema_slow": None, "bullish_cross": None}
    zs = [logit(p, half_tick=half_tick) for p in prices]
    f = ema_series(zs, fast)
    s = ema_series(zs, slow)
    ef, es = f[-1], s[-1]
    prev_f, prev_s = f[-2], s[-2]
    cross = False
    if ef is not None and es is not None and prev_f is not None and prev_s is not None:
        cross = prev_f <= prev_s and ef > es
    return {
        "ema_fast": ef,
        "ema_slow": es,
        "bullish_cross": cross if ef is not None and es is not None else None,
        "bullish_stack": (ef > es) if ef is not None and es is not None else None,
    }


def robust_logit_bands(
    prices: Sequence[float],
    *,
    window: int = 48,
    k: float = 2.5,
    half_tick: float = DEFAULT_HALF_TICK,
) -> dict[str, float | None]:
    """Median ± k·MAD bands in logit space."""
    if len(prices) < window:
        return {"mid": None, "upper": None, "lower": None, "mad": None}
    zs = [logit(p, half_tick=half_tick) for p in prices[-window:]]
    zs_sorted = sorted(zs)
    mid = zs_sorted[len(zs_sorted) // 2]
    mad = sorted(abs(z - mid) for z in zs)[len(zs) // 2]
    # Consistency constant ≈ 1.4826 for Gaussian; keep raw MAD scale for sweeps
    return {
        "mid": mid,
        "upper": mid + k * mad,
        "lower": mid - k * mad,
        "mad": mad,
    }


def attach_logit_technicals(
    df: pl.DataFrame,
    *,
    price_col: str = "price",
    group_col: str = "token_id",
    rsi_period: int = 14,
    ema_fast: int = 5,
    ema_slow: int = 20,
    band_window: int = 48,
    band_k: float = 2.5,
    half_tick: float = DEFAULT_HALF_TICK,
) -> pl.DataFrame:
    """
    Attach arcsine + rolling logit RSI/EMA/MAD columns when token panels exist.

    Uses map_groups for correctness on short panels; null when history insufficient.
    """
    if df.is_empty() or price_col not in df.columns:
        return df
    out = attach_arcsine(df, price_col=price_col)
    if group_col not in out.columns:
        return out

    def _per_group(g: pl.DataFrame) -> pl.DataFrame:
        prices = g[price_col].to_list()
        rsi_vals: list[float | None] = []
        ef_vals: list[float | None] = []
        es_vals: list[float | None] = []
        cross_vals: list[bool | None] = []
        upper_vals: list[float | None] = []
        lower_vals: list[float | None] = []
        for i in range(len(prices)):
            hist = prices[: i + 1]
            rsi_vals.append(logit_rsi(hist, period=rsi_period, half_tick=half_tick))
            cross = logit_ema_cross(
                hist, fast=ema_fast, slow=ema_slow, half_tick=half_tick
            )
            ef_vals.append(cross["ema_fast"])  # type: ignore[arg-type]
            es_vals.append(cross["ema_slow"])  # type: ignore[arg-type]
            cross_vals.append(cross["bullish_cross"])  # type: ignore[arg-type]
            bands = robust_logit_bands(
                hist, window=band_window, k=band_k, half_tick=half_tick
            )
            upper_vals.append(bands["upper"])
            lower_vals.append(bands["lower"])
        return g.with_columns(
            [
                pl.Series("logit_rsi", rsi_vals),
                pl.Series("logit_ema_fast", ef_vals),
                pl.Series("logit_ema_slow", es_vals),
                pl.Series("logit_ema_bullish_cross", cross_vals),
                pl.Series("logit_mad_upper", upper_vals),
                pl.Series("logit_mad_lower", lower_vals),
            ]
        )

    sort_cols = [group_col]
    if "traded_at" in out.columns:
        sort_cols.append("traded_at")
    return out.sort(sort_cols).group_by(group_col, maintain_order=True).map_groups(_per_group)
