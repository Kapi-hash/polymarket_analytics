"""Leakage-safe validation scaffolding: purged walk-forward, calibrators, DSR/PBO/FDR."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Sequence

import polars as pl


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: str
    train_end: str  # exclusive
    test_start: str
    test_end: str  # exclusive
    embargo_start: str
    embargo_end: str


@dataclass(frozen=True)
class MultipleTestingReport:
    n_trials: int
    best_sharpe: float
    dsr: float | None
    pbo_proxy: float | None
    fdr_alpha: float
    n_discoveries_bh: int
    neighborhood_stability: float | None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_utc(dt: datetime | str) -> datetime:
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    text = str(dt).strip().replace("Z", "+00:00")
    if "T" in text:
        parsed = datetime.fromisoformat(text)
    else:
        y, m, d = (int(x) for x in text.split("-", 2))
        parsed = datetime(y, m, d, tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def event_groups(df: pl.DataFrame, event_col: str = "condition_id") -> pl.Series:
    if event_col not in df.columns:
        raise KeyError(f"Missing event column: {event_col}")
    return df[event_col]


def purged_walk_forward_folds(
    *,
    start: datetime | str,
    end: datetime | str,
    n_folds: int = 3,
    train_days: int = 30,
    test_days: int = 7,
    embargo_days: int = 2,
) -> list[WalkForwardFold]:
    """
    Build time folds with an embargo gap between train and test.

    Groups by event should additionally drop overlapping condition_ids
    (see ``mask_event_leakage``).
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    t0 = _to_utc(start)
    t_end = _to_utc(end)
    folds: list[WalkForwardFold] = []
    cursor = t0
    for i in range(n_folds):
        train_start = cursor
        train_end = train_start + timedelta(days=train_days)
        embargo_start = train_end
        embargo_end = embargo_start + timedelta(days=embargo_days)
        test_start = embargo_end
        test_end = test_start + timedelta(days=test_days)
        if test_end > t_end + timedelta(days=1):
            break
        folds.append(
            WalkForwardFold(
                fold_id=i,
                train_start=train_start.date().isoformat(),
                train_end=train_end.date().isoformat(),
                test_start=test_start.date().isoformat(),
                test_end=test_end.date().isoformat(),
                embargo_start=embargo_start.date().isoformat(),
                embargo_end=embargo_end.date().isoformat(),
            )
        )
        cursor = cursor + timedelta(days=test_days)
    return folds


def mask_event_leakage(
    df: pl.DataFrame,
    train_events: set[str],
    *,
    event_col: str = "condition_id",
    mode: str = "drop_train_events_from_test",
) -> pl.DataFrame:
    """Remove test rows whose event appeared in train (grouped purge)."""
    if df.is_empty() or event_col not in df.columns:
        return df
    if mode != "drop_train_events_from_test":
        raise ValueError(f"Unknown mode: {mode}")
    return df.filter(~pl.col(event_col).is_in(list(train_events)))


def fit_bin_calibrator(
    prices: Sequence[float],
    outcomes: Sequence[int | bool],
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Fit a simple histogram calibrator on TRAIN only.

    Returns bin edges + empirical frequencies. Apply via ``apply_bin_calibrator``.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    edges = [i / n_bins for i in range(n_bins + 1)]
    counts = [0] * n_bins
    wins = [0] * n_bins
    for p, y in zip(prices, outcomes):
        x = min(max(float(p), 0.0), 1.0 - 1e-12)
        idx = min(int(x * n_bins), n_bins - 1)
        counts[idx] += 1
        wins[idx] += int(bool(y))
    freqs = [ (wins[i] / counts[i]) if counts[i] else None for i in range(n_bins) ]
    return {"n_bins": n_bins, "edges": edges, "counts": counts, "freqs": freqs}


def apply_bin_calibrator(price: float, calibrator: dict[str, Any]) -> float | None:
    n_bins = int(calibrator["n_bins"])
    x = min(max(float(price), 0.0), 1.0 - 1e-12)
    idx = min(int(x * n_bins), n_bins - 1)
    return calibrator["freqs"][idx]


def calibration_residuals(
    prices: Sequence[float],
    outcomes: Sequence[int | bool],
    calibrator: dict[str, Any],
) -> list[float | None]:
    """outcome - calibrated_p for each row (None if bin empty)."""
    out: list[float | None] = []
    for p, y in zip(prices, outcomes):
        cal = apply_bin_calibrator(p, calibrator)
        if cal is None:
            out.append(None)
        else:
            out.append(float(bool(y)) - float(cal))
    return out


def deflated_sharpe_ratio(
    sharpe: float,
    *,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float | None:
    """
    Bailey & López de Prado DSR scaffolding (simplified).

    Returns approximate probability that observed SR is significant vs max of n_trials.
    """
    if n_obs < 2 or n_trials < 1:
        return None
    # Expected max SR under null (rough Euler-Mascheroni approx)
    em = 0.5772156649
    sr0 = ((1 - em) * math.pow(math.pi / 6.0 / max(n_trials, 1), 0.5)
           + em * math.pow(2.0 * math.log(max(n_trials, 2)), 0.5))
    # SR variance adjustment
    se = math.sqrt(
        (1.0 + 0.5 * sharpe * sharpe - skew * sharpe + (kurt - 3.0) / 4.0 * sharpe * sharpe)
        / max(n_obs - 1, 1)
    )
    if se <= 0:
        return None
    z = (sharpe - sr0) / se
    # Φ(z)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def pbo_proxy(is_sharpes: Sequence[float], oos_sharpes: Sequence[float]) -> float | None:
    """
    Probability of backtest overfitting proxy:
    fraction of trials where IS rank beats median but OOS is below median.
    """
    if len(is_sharpes) < 2 or len(is_sharpes) != len(oos_sharpes):
        return None
    n = len(is_sharpes)
    is_med = sorted(is_sharpes)[n // 2]
    oos_med = sorted(oos_sharpes)[n // 2]
    bad = sum(
        1
        for a, b in zip(is_sharpes, oos_sharpes)
        if a >= is_med and b < oos_med
    )
    return bad / n


def benjamini_hochberg(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> list[bool]:
    """BH FDR control; returns reject flags aligned with input order."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    max_k = -1
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= (rank / m) * alpha:
            max_k = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= max_k:
            reject[idx] = True
    return reject


def neighborhood_stability(
    scores: dict[str, float],
    center_key: str,
    neighbors: Sequence[str],
) -> float | None:
    """Fraction of neighbor configs with same sign as center score."""
    if center_key not in scores or not neighbors:
        return None
    c = scores[center_key]
    if c == 0:
        return None
    same = sum(1 for k in neighbors if k in scores and scores[k] * c > 0)
    return same / len(neighbors)


def multiple_testing_controls(
    *,
    sharpes: Sequence[float],
    oos_sharpes: Sequence[float] | None = None,
    n_obs: int = 100,
    fdr_alpha: float = 0.05,
    p_values: Sequence[float] | None = None,
    scores: dict[str, float] | None = None,
    center_key: str | None = None,
    neighbors: Sequence[str] | None = None,
) -> MultipleTestingReport:
    best = max(sharpes) if sharpes else 0.0
    dsr = deflated_sharpe_ratio(best, n_obs=n_obs, n_trials=max(len(sharpes), 1))
    pbo = pbo_proxy(list(sharpes), list(oos_sharpes)) if oos_sharpes else None
    n_disc = 0
    if p_values:
        n_disc = sum(benjamini_hochberg(p_values, alpha=fdr_alpha))
    stab = None
    if scores and center_key and neighbors:
        stab = neighborhood_stability(scores, center_key, neighbors)
    return MultipleTestingReport(
        n_trials=len(sharpes),
        best_sharpe=best,
        dsr=dsr,
        pbo_proxy=pbo,
        fdr_alpha=fdr_alpha,
        n_discoveries_bh=n_disc,
        neighborhood_stability=stab,
        notes="Scaffolding only — not a claim of alpha",
    )


def run_grouped_purged_validation(
    df: pl.DataFrame,
    evaluate: Callable[[pl.DataFrame, pl.DataFrame], dict[str, Any]],
    *,
    event_col: str = "condition_id",
    time_col: str = "traded_at",
    start: str,
    end: str,
    n_folds: int = 2,
    train_days: int = 14,
    test_days: int = 7,
    embargo_days: int = 2,
) -> dict[str, Any]:
    """
    Small representative walk-forward runner.

    ``evaluate(train_df, test_df)`` must fit any calibrators on train only.
    """
    folds = purged_walk_forward_folds(
        start=start,
        end=end,
        n_folds=n_folds,
        train_days=train_days,
        test_days=test_days,
        embargo_days=embargo_days,
    )
    results: list[dict[str, Any]] = []
    for fold in folds:
        train = df.filter(
            (pl.col(time_col) >= pl.lit(fold.train_start).str.to_datetime(time_zone="UTC"))
            & (pl.col(time_col) < pl.lit(fold.train_end).str.to_datetime(time_zone="UTC"))
        )
        test = df.filter(
            (pl.col(time_col) >= pl.lit(fold.test_start).str.to_datetime(time_zone="UTC"))
            & (pl.col(time_col) < pl.lit(fold.test_end).str.to_datetime(time_zone="UTC"))
        )
        if event_col in train.columns:
            train_events = set(train[event_col].drop_nulls().to_list())
            test = mask_event_leakage(test, train_events, event_col=event_col)
        metrics = evaluate(train, test)
        results.append({"fold": asdict(fold), "metrics": metrics})
    return {"n_folds": len(results), "folds": results}
