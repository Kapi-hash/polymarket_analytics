"""Validated statistics for outcome research — no invented p-values."""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

import polars as pl


def event_clustered_returns(
    trades_df: pl.DataFrame,
    *,
    pnl_col: str = "pnl",
    event_col: str = "condition_id",
) -> pl.DataFrame:
    """
    Aggregate per-trade PnL to event-level returns (one row per event).

    Requires ``pnl_col`` and ``event_col`` on ``trades_df``.
    """
    if trades_df.is_empty():
        return pl.DataFrame(schema={event_col: pl.Utf8, "event_pnl": pl.Float64, "n_trades": pl.Int64})
    if pnl_col not in trades_df.columns:
        raise KeyError(f"Missing pnl column: {pnl_col}")
    if event_col not in trades_df.columns:
        raise KeyError(f"Missing event column: {event_col}")

    return (
        trades_df.group_by(event_col)
        .agg(
            pl.col(pnl_col).sum().alias("event_pnl"),
            pl.len().alias("n_trades"),
        )
        .sort(event_col)
    )


def cluster_bootstrap_ci(
    event_pnls: Sequence[float],
    *,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Bootstrap CI on mean event PnL (cluster = event).

    Returns mean, lo, hi, p_positive.
    """
    vals = [float(x) for x in event_pnls if x is not None and not math.isnan(float(x))]
    n = len(vals)
    if n == 0:
        return {"status": "unavailable", "reason": "empty event_pnls", "n_events": 0}

    mean = sum(vals) / n
    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(max(n_boot, 1)):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo_idx = int((alpha / 2.0) * len(boot_means))
    hi_idx = int((1.0 - alpha / 2.0) * len(boot_means)) - 1
    lo_idx = max(0, min(lo_idx, len(boot_means) - 1))
    hi_idx = max(0, min(hi_idx, len(boot_means) - 1))
    p_pos = sum(1 for m in boot_means if m > 0) / len(boot_means)

    return {
        "status": "ok",
        "mean": mean,
        "lo": boot_means[lo_idx],
        "hi": boot_means[hi_idx],
        "p_positive": p_pos,
        "n_events": n,
        "n_boot": n_boot,
    }


def deflated_sharpe_valid(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> dict[str, Any]:
    """
    Bailey / López de Prado deflated Sharpe ratio.

    Returns None probability when insufficient data.
    """
    if n_obs < 2 or n_trials < 1:
        return {"status": "unavailable", "reason": "insufficient n_obs or n_trials", "dsr": None}

    em = 0.5772156649
    sr0 = (1 - em) * math.pow(math.pi / 6.0 / max(n_trials, 1), 0.5) + em * math.pow(
        2.0 * math.log(max(n_trials, 2)), 0.5
    )
    se = math.sqrt(
        (1.0 + 0.5 * sharpe * sharpe - skew * sharpe + (kurt - 3.0) / 4.0 * sharpe * sharpe)
        / max(n_obs - 1, 1)
    )
    if se <= 0:
        return {"status": "unavailable", "reason": "non-positive standard error", "dsr": None}

    z = (sharpe - sr0) / se
    dsr = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return {"status": "ok", "dsr": dsr, "sr0": sr0, "z": z}


def pbo_from_matrices(
    is_sharpes: Sequence[Sequence[float]],
    oos_sharpes: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """
    Probability of backtest overfitting.

    Uses combinatorial CSCV when partition count is small (<= 8);
    otherwise a rank-based proxy. Never returns fake zeros without data.
    """
    if not is_sharpes or not oos_sharpes:
        return {"status": "unavailable", "reason": "empty sharpe matrices"}
    if len(is_sharpes) != len(oos_sharpes):
        return {"status": "unavailable", "reason": "matrix row count mismatch"}

    n_partitions = len(is_sharpes)
    n_trials = len(is_sharpes[0]) if is_sharpes else 0
    if n_trials < 2:
        return {"status": "unavailable", "reason": "need at least 2 trials"}

    for row_is, row_oos in zip(is_sharpes, oos_sharpes):
        if len(row_is) != n_trials or len(row_oos) != n_trials:
            return {"status": "unavailable", "reason": "inconsistent trial width"}

    # Combinatorial: for each trial, count partitions where IS best but OOS below median
    if n_partitions <= 8:
        oos_medians = [sorted(row)[n_trials // 2] for row in oos_sharpes]
        logit_vals: list[float] = []
        for j in range(n_trials):
            is_vals = [is_sharpes[i][j] for i in range(n_partitions)]
            oos_vals = [oos_sharpes[i][j] for i in range(n_partitions)]
            best_is = max(is_vals)
            best_idx = is_vals.index(best_is)
            oos_at_best = oos_vals[best_idx]
            below = sum(1 for v in oos_vals if v < oos_medians[best_idx])
            logit_vals.append(below / n_partitions)
        pbo = sum(1 for v in logit_vals if v > 0.5) / n_trials
        return {"status": "ok", "pbo": pbo, "method": "combinatorial_cscv", "n_partitions": n_partitions}

    # Proxy for larger matrices
    bad = 0
    for j in range(n_trials):
        is_col = [is_sharpes[i][j] for i in range(n_partitions)]
        oos_col = [oos_sharpes[i][j] for i in range(n_partitions)]
        is_med = sorted(is_col)[n_partitions // 2]
        oos_med = sorted(oos_col)[n_partitions // 2]
        if max(is_col) >= is_med and min(oos_col) < oos_med:
            bad += 1
    return {
        "status": "ok",
        "pbo": bad / n_trials,
        "method": "rank_proxy",
        "n_partitions": n_partitions,
    }


def bh_fdr(
    p_values: Sequence[float] | None,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Benjamini-Hochberg FDR; unavailable when no valid p-values."""
    if p_values is None or len(p_values) == 0:
        return {"status": "unavailable", "reason": "no p-values supplied"}

    cleaned: list[tuple[int, float]] = []
    for i, p in enumerate(p_values):
        if p is None or math.isnan(float(p)):
            continue
        pv = float(p)
        if pv < 0.0 or pv > 1.0:
            return {"status": "unavailable", "reason": f"invalid p-value at index {i}"}
        cleaned.append((i, pv))

    if not cleaned:
        return {"status": "unavailable", "reason": "no valid p-values"}

    m = len(cleaned)
    order = sorted(cleaned, key=lambda x: x[1])
    reject_indices: set[int] = set()
    max_k = -1
    for rank, (idx, p) in enumerate(order, start=1):
        if p <= (rank / m) * alpha:
            max_k = rank
    for rank, (idx, _) in enumerate(order, start=1):
        if rank <= max_k:
            reject_indices.add(idx)

    reject_flags = [i in reject_indices for i, _ in cleaned]
    return {
        "status": "ok",
        "n_tests": m,
        "n_discoveries": sum(reject_flags),
        "reject_flags": reject_flags,
        "alpha": alpha,
    }


def leave_five_largest_out(event_pnls: Sequence[float]) -> dict[str, Any]:
    """Sensitivity: mean with five largest event PnLs removed."""
    vals = sorted([float(x) for x in event_pnls], reverse=True)
    if len(vals) <= 5:
        return {
            "status": "unavailable",
            "reason": "fewer than 6 events for leave-5-out",
            "n_events": len(vals),
        }
    trimmed = vals[5:]
    return {
        "status": "ok",
        "mean_full": sum(vals) / len(vals),
        "mean_trimmed": sum(trimmed) / len(trimmed),
        "n_events_full": len(vals),
        "n_events_trimmed": len(trimmed),
        "removed_sum": sum(vals[:5]),
    }
