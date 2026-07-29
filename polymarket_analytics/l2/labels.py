"""Forward labels at multiple horizons with gap-aware skipping."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from polymarket_analytics.research.logit import logit

LABEL_HORIZONS_SEC: tuple[int, ...] = (5, 30, 60, 300, 900, 3600, 14400)


@dataclass
class ForwardLabel:
    horizon_sec: int
    mid_return: float | None = None
    logit_return: float | None = None
    taker_return: float | None = None
    maker_return_lower: float | None = None
    maker_return_upper: float | None = None
    mfe: float | None = None
    mae: float | None = None
    time_to_tp_sec: float | None = None
    time_to_sl_sec: float | None = None
    first_hit: str | None = None
    post_entry_adverse_selection: float | None = None
    skipped: bool = False
    skip_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _crosses_gap(times: Sequence[float], t0: float, t1: float, gaps: Sequence[tuple[float, float]]) -> bool:
    for gs, ge in gaps:
        if gs <= t0 <= ge or gs <= t1 <= ge:
            return True
        if t0 <= gs and t1 >= ge:
            return True
    return False


def compute_forward_labels(
    *,
    entry_time: float,
    entry_mid: float,
    entry_side: str,
    series: Sequence[dict[str, Any]],
    horizons_sec: Sequence[int] | None = None,
    gaps: Sequence[tuple[float, float]] | None = None,
    tp_pct: float | None = 0.05,
    sl_pct: float | None = 0.03,
    taker_entry_price: float | None = None,
    maker_bounds: tuple[float, float] | None = None,
    invalid_intervals: Sequence[tuple[float, float]] | None = None,
) -> list[ForwardLabel]:
    """Compute forward labels; skip horizons crossing gaps or invalid book intervals."""
    horizons = tuple(horizons_sec or LABEL_HORIZONS_SEC)
    gap_list = list(gaps or [])
    invalid = list(invalid_intervals or [])
    labels: list[ForwardLabel] = []

    sorted_series = sorted(series, key=lambda r: float(r.get("time", r.get("exchange_time", 0))))

    for h in horizons:
        label = ForwardLabel(horizon_sec=h)
        end_t = entry_time + h
        if _crosses_gap([], entry_time, end_t, gap_list):
            label.skipped = True
            label.skip_reason = "gap"
            labels.append(label)
            continue
        for ivs, ive in invalid:
            if ivs <= entry_time <= ive or ivs <= end_t <= ive:
                label.skipped = True
                label.skip_reason = "invalid_book"
                labels.append(label)
                break
        else:
            window = [r for r in sorted_series if entry_time < float(r.get("time", r.get("exchange_time", 0))) <= end_t]
            if not window:
                label.skipped = True
                label.skip_reason = "no_data"
                labels.append(label)
                continue

            mids = [float(r.get("mid", r.get("best_bid", 0))) for r in window if r.get("mid") is not None]
            if not mids:
                label.skipped = True
                label.skip_reason = "no_mid"
                labels.append(label)
                continue

            exit_mid = mids[-1]
            raw_ret = (exit_mid - entry_mid) / entry_mid if entry_mid > 0 else 0.0
            if entry_side == "short":
                raw_ret = -raw_ret
            label.mid_return = raw_ret
            label.logit_return = logit(min(max(exit_mid, 0.005), 0.995)) - logit(min(max(entry_mid, 0.005), 0.995))

            path_rets = [(m - entry_mid) / entry_mid for m in mids]
            if entry_side == "short":
                path_rets = [-r for r in path_rets]
            label.mfe = max(path_rets)
            label.mae = min(path_rets)

            if taker_entry_price is not None and entry_mid > 0:
                label.taker_return = (exit_mid - taker_entry_price) / taker_entry_price
                if entry_side == "short":
                    label.taker_return = -label.taker_return
                label.post_entry_adverse_selection = label.taker_return - label.mid_return

            if maker_bounds is not None:
                lo, hi = maker_bounds
                label.maker_return_lower = label.mid_return * lo
                label.maker_return_upper = label.mid_return * hi

            tp_t = sl_t = None
            first = None
            for i, r in enumerate(path_rets):
                t = float(window[i].get("time", window[i].get("exchange_time", entry_time)))
                if tp_pct is not None and r >= tp_pct and tp_t is None:
                    tp_t = t - entry_time
                    if first is None:
                        first = "tp"
                if sl_pct is not None and r <= -sl_pct and sl_t is None:
                    sl_t = t - entry_time
                    if first is None:
                        first = "sl"
            label.time_to_tp_sec = tp_t
            label.time_to_sl_sec = sl_t
            label.first_hit = first
            labels.append(label)

    return labels
