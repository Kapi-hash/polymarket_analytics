"""Pure L2 microstructure feature formulas."""

from __future__ import annotations

import math
from typing import Sequence

from polymarket_analytics.research.logit import clamp_prob, logit


def abs_spread(best_bid: float | None, best_ask: float | None) -> float | None:
    if best_bid is None or best_ask is None:
        return None
    return float(best_ask) - float(best_bid)


def rel_spread(best_bid: float | None, best_ask: float | None, mid: float | None = None) -> float | None:
    spr = abs_spread(best_bid, best_ask)
    if spr is None:
        return None
    m = mid if mid is not None else ((float(best_bid) + float(best_ask)) / 2.0)
    if m <= 0:
        return None
    return spr / m


def logit_spread(best_bid: float | None, best_ask: float | None) -> float | None:
    if best_bid is None or best_ask is None:
        return None
    return logit(float(best_ask)) - logit(float(best_bid))


def multi_level_imbalance(bid_sizes: Sequence[float], ask_sizes: Sequence[float]) -> float:
    n = min(len(bid_sizes), len(ask_sizes))
    if n == 0:
        return 0.0
    num = sum(float(bid_sizes[i]) - float(ask_sizes[i]) for i in range(n))
    den = sum(float(bid_sizes[i]) + float(ask_sizes[i]) for i in range(n))
    return num / den if den > 0 else 0.0


def distance_weighted_imbalance(
    bid_levels: Sequence[tuple[float, float]],
    ask_levels: Sequence[tuple[float, float]],
    *,
    mid: float,
    decay: float = 0.5,
) -> float:
    num = 0.0
    den = 0.0
    for p, s in bid_levels:
        w = math.exp(-decay * abs(float(mid) - float(p)))
        num += w * float(s)
        den += w * float(s)
    for p, s in ask_levels:
        w = math.exp(-decay * abs(float(p) - float(mid)))
        num -= w * float(s)
        den += w * float(s)
    return num / den if den > 0 else 0.0


def book_slope(levels: Sequence[tuple[float, float]], *, side: str) -> float | None:
    if len(levels) < 2:
        return None
    xs = [float(p) for p, _ in levels[:5]]
    ys = [float(s) for _, s in levels[:5]]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(len(xs)))
    den = sum((xs[i] - x_mean) ** 2 for i in range(len(xs)))
    if den == 0:
        return None
    slope = num / den
    return -slope if side == "bid" else slope


def depth_concentration(sizes: Sequence[float]) -> float | None:
    total = sum(float(s) for s in sizes)
    if total <= 0:
        return None
    top = float(sizes[0]) if sizes else 0.0
    return top / total


def depth_entropy(sizes: Sequence[float]) -> float | None:
    total = sum(float(s) for s in sizes if float(s) > 0)
    if total <= 0:
        return None
    ent = 0.0
    for s in sizes:
        p = float(s) / total
        if p > 0:
            ent -= p * math.log(p)
    return ent


def tob_vs_deep_divergence(tob_imb: float, deep_imb: float) -> float:
    return float(tob_imb) - float(deep_imb)


def liquidity_wall_score(levels: Sequence[tuple[float, float]], *, mid: float, threshold: float = 3.0) -> float:
    if not levels:
        return 0.0
    sizes = [float(s) for _, s in levels]
    avg = sum(sizes) / len(sizes) if sizes else 0.0
    if avg <= 0:
        return 0.0
    return max(float(s) / avg for s in sizes) if sizes else 0.0


def boundary_adjusted_depth(depth: float, prob: float) -> float:
    p = clamp_prob(prob)
    return float(depth) * p * (1.0 - p)


def ofi_delta(
    bid_size: float,
    ask_size: float,
    prev_bid_size: float,
    prev_ask_size: float,
) -> float:
    return (float(bid_size) - float(prev_bid_size)) - (float(ask_size) - float(prev_ask_size))


def signed_trade_imbalance(buy_vol: float, sell_vol: float) -> float:
    den = float(buy_vol) + float(sell_vol)
    if den <= 0:
        return 0.0
    return (float(buy_vol) - float(sell_vol)) / den


def arrival_intensity(count: int, window_sec: float) -> float:
    if window_sec <= 0:
        return 0.0
    return float(count) / float(window_sec)


def cancel_trade_ratio(cancel_count: int, trade_count: int) -> float | None:
    if trade_count <= 0:
        return None
    return float(cancel_count) / float(trade_count)


def refill_rate(depth_before: float, depth_after: float, dt_sec: float) -> float | None:
    if dt_sec <= 0 or depth_before <= 0:
        return None
    return max(0.0, (float(depth_after) - float(depth_before)) / float(depth_before)) / dt_sec


def depletion_velocity(depth_before: float, depth_after: float, dt_sec: float) -> float | None:
    if dt_sec <= 0:
        return None
    return (float(depth_before) - float(depth_after)) / dt_sec


def flow_autocorr(series: Sequence[float], lag: int = 1) -> float | None:
    if len(series) <= lag:
        return None
    x = [float(v) for v in series]
    mean = sum(x) / len(x)
    num = sum((x[i] - mean) * (x[i - lag] - mean) for i in range(lag, len(x)))
    den = sum((v - mean) ** 2 for v in x)
    return num / den if den > 0 else None


def vpin_proxy(buy_vol: float, sell_vol: float, total_vol: float) -> float | None:
    if total_vol <= 0:
        return None
    return abs(float(buy_vol) - float(sell_vol)) / float(total_vol)


def kyle_lambda_proxy(price_change: float, signed_volume: float) -> float | None:
    if signed_volume == 0:
        return None
    return float(price_change) / float(signed_volume)


def amihud_proxy(abs_return: float, dollar_volume: float) -> float | None:
    if dollar_volume <= 0:
        return None
    return abs(float(abs_return)) / float(dollar_volume)


def spread_recovery_time(spreads: Sequence[float], shock_idx: int, *, baseline: float | None = None) -> float | None:
    if shock_idx < 0 or shock_idx >= len(spreads):
        return None
    base = baseline if baseline is not None else float(spreads[max(0, shock_idx - 1)])
    for i in range(shock_idx + 1, len(spreads)):
        if float(spreads[i]) <= base * 1.05:
            return float(i - shock_idx)
    return None


def imbalance_mean_reversion_time(imb_series: Sequence[float], shock_idx: int) -> float | None:
    if shock_idx < 0 or shock_idx >= len(imb_series):
        return None
    target = 0.0
    shock = float(imb_series[shock_idx])
    for i in range(shock_idx + 1, len(imb_series)):
        if abs(float(imb_series[i]) - target) <= abs(shock) * 0.5:
            return float(i - shock_idx)
    return None


def yes_no_complement_deviation(yes_mid: float, no_mid: float) -> float:
    return float(yes_mid) + float(no_mid) - 1.0


def complete_set_residual(yes_mid: float, no_mid: float) -> float:
    return 1.0 - (float(yes_mid) + float(no_mid))


def ttr_decay_factor(ttr_hours: float, *, half_life_hours: float = 168.0) -> float:
    if ttr_hours <= 0:
        return 1.0
    return math.exp(-math.log(2) * float(ttr_hours) / float(half_life_hours))


def resolution_proximity(ttr_hours: float, *, max_hours: float = 720.0) -> float:
    if ttr_hours <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - float(ttr_hours) / float(max_hours)))


def vol_scaled_by_p_var(prob: float, vol: float) -> float:
    p = clamp_prob(prob)
    return float(vol) * p * (1.0 - p)


def imbalance_x_p_var(imbalance: float, prob: float) -> float:
    p = clamp_prob(prob)
    return float(imbalance) * p * (1.0 - p)


def spread_normalized_upside(spread: float, prob: float, *, side: str = "buy") -> float | None:
    p = clamp_prob(prob)
    upside = (1.0 - p) if side == "buy" else p
    if upside <= 0:
        return None
    return float(spread) / upside


def classify_cancel_vs_trade(prev_size: float, new_size: float, trade_size: float) -> str:
    delta = float(new_size) - float(prev_size)
    if trade_size > 0 and delta <= -trade_size * 0.5:
        return "trade"
    if delta < 0:
        return "cancel"
    if delta > 0:
        return "add"
    return "unchanged"
