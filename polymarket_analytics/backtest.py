"""Phase 3: edge finder (grid scan) and strategy backtester."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import polars as pl

from polymarket_analytics.schema import PRICE_BUCKET_LABELS
from polymarket_analytics.store import bootstrap_warehouse, connect

MomentumSign = Literal["any", "pos", "neg"]

DEFAULT_SPIKE_THRESHOLDS: tuple[float | None, ...] = (None, 1.5, 2.0, 3.0)
DEFAULT_TTR_BOUNDS: tuple[float | None, ...] = (None, 24.0, 48.0, 72.0)
DEFAULT_WHALE_THRESHOLDS: tuple[float | None, ...] = (None, 2.0, 3.0)
DEFAULT_MOMENTUM_SIGNS: tuple[MomentumSign, ...] = ("any", "pos", "neg")


@dataclass(frozen=True)
class StrategyParams:
    """Entry filter for a long-token (buy-at-price) strategy."""

    price_bucket: str | None = None
    min_volume_spike: float | None = None
    min_whale_ratio: float | None = None
    require_price_volume_divergence: bool = False
    momentum_1h: MomentumSign = "any"
    momentum_6h: MomentumSign = "any"
    max_time_to_resolution_hours: float | None = None
    side: str | None = "BUY"  # optional fill-side filter

    def label(self) -> str:
        parts: list[str] = []
        if self.price_bucket:
            parts.append(f"bucket={self.price_bucket}")
        if self.min_volume_spike is not None:
            parts.append(f"spike>{self.min_volume_spike:g}")
        if self.min_whale_ratio is not None:
            parts.append(f"whale>{self.min_whale_ratio:g}")
        if self.require_price_volume_divergence:
            parts.append("pvd=1")
        if self.momentum_1h != "any":
            parts.append(f"mom1h={self.momentum_1h}")
        if self.momentum_6h != "any":
            parts.append(f"mom6h={self.momentum_6h}")
        if self.max_time_to_resolution_hours is not None:
            parts.append(f"ttr<{self.max_time_to_resolution_hours:g}h")
        if self.side:
            parts.append(f"side={self.side}")
        return " AND ".join(parts) if parts else "all"


@dataclass(frozen=True)
class EdgeStats:
    """Slice-level edge summary."""

    n: int
    empirical_win_rate: float
    implied_win_rate: float
    ev_pct: float
    params: dict[str, Any]
    label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestResult:
    """Simulated strategy performance."""

    n: int
    win_rate: float
    implied_win_rate: float
    ev_pct: float
    total_pnl: float
    avg_pnl: float
    sharpe: float
    max_drawdown: float
    params: dict[str, Any]
    label: str
    equity_curve: list[float]

    def to_dict(self, *, include_equity: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_equity:
            d.pop("equity_curve", None)
        return d


def load_trade_features(
    warehouse_path: Path | str,
    parquet_dir: Path | str | None = None,
) -> pl.DataFrame:
    """Load ``v_trade_features`` via DuckDB (refreshes views if parquet_dir given)."""
    warehouse_path = Path(warehouse_path)
    if parquet_dir is not None:
        bootstrap_warehouse(Path(parquet_dir), warehouse_path)

    con = connect(warehouse_path)
    try:
        return con.execute("SELECT * FROM v_trade_features").pl()
    finally:
        con.close()


def _momentum_expr(col: str, sign: MomentumSign) -> pl.Expr | None:
    if sign == "any":
        return None
    if sign == "pos":
        return pl.col(col) > 0
    if sign == "neg":
        return pl.col(col) < 0
    raise ValueError(f"Unknown momentum sign: {sign}")


def apply_strategy_filter(df: pl.DataFrame, params: StrategyParams) -> pl.DataFrame:
    """Vectorized Polars filter for a parameter set."""
    if df.is_empty():
        return df

    exprs: list[pl.Expr] = []
    if params.price_bucket is not None:
        exprs.append(pl.col("price_bucket") == params.price_bucket)
    if params.min_volume_spike is not None:
        exprs.append(
            pl.col("volume_spike_1h_24h").is_not_null()
            & (pl.col("volume_spike_1h_24h") > params.min_volume_spike)
        )
    if params.min_whale_ratio is not None:
        if "whale_ratio" in df.columns:
            exprs.append(
                pl.col("whale_ratio").is_not_null()
                & (pl.col("whale_ratio") > params.min_whale_ratio)
            )
        else:
            exprs.append(pl.lit(False))
    if params.require_price_volume_divergence:
        if "price_volume_divergence" in df.columns:
            exprs.append(pl.col("price_volume_divergence") == True)  # noqa: E712
        else:
            exprs.append(pl.lit(False))
    m1 = _momentum_expr("momentum_1h", params.momentum_1h)
    if m1 is not None:
        exprs.append(pl.col("momentum_1h").is_not_null() & m1)
    m6 = _momentum_expr("momentum_6h", params.momentum_6h)
    if m6 is not None:
        exprs.append(pl.col("momentum_6h").is_not_null() & m6)
    if params.max_time_to_resolution_hours is not None:
        exprs.append(
            pl.col("time_to_resolution_hours").is_not_null()
            & (pl.col("time_to_resolution_hours") < params.max_time_to_resolution_hours)
        )
    if params.side is not None and "side" in df.columns:
        exprs.append(pl.col("side").cast(pl.Utf8).str.to_uppercase() == params.side.upper())

    if not exprs:
        return df

    mask = exprs[0]
    for e in exprs[1:]:
        mask = mask & e
    return df.filter(mask)


def compute_edge_stats(df: pl.DataFrame, params: StrategyParams) -> EdgeStats:
    """
    N, empirical win rate, implied win rate (avg price), EV%.

    Long $1 payoff token bought at ``price``:
      per-share PnL = token_won - price
      EV% = 100 * mean(token_won - price) = 100 * (empirical - implied)
    """
    n = df.height
    if n == 0:
        return EdgeStats(
            n=0,
            empirical_win_rate=0.0,
            implied_win_rate=0.0,
            ev_pct=0.0,
            params=asdict(params),
            label=params.label(),
        )

    won = df["token_won"].cast(pl.Float64)
    price = df["price"].cast(pl.Float64)
    empirical = float(won.mean())
    implied = float(price.mean())
    ev_pct = 100.0 * (empirical - implied)
    return EdgeStats(
        n=n,
        empirical_win_rate=empirical,
        implied_win_rate=implied,
        ev_pct=ev_pct,
        params=asdict(params),
        label=params.label(),
    )


def _max_drawdown(equity: Sequence[float]) -> float:
    """Max peak-to-trough drawdown on cumulative PnL (absolute $)."""
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def _sharpe_ratio(returns: pl.Series) -> float:
    """Sample Sharpe on per-trade ROI (mean/std); 0 if undefined."""
    if returns.len() < 2:
        return 0.0
    mu = returns.mean()
    sigma = returns.std()
    if mu is None or sigma is None or sigma == 0:
        return 0.0
    return float(mu / sigma * (returns.len() ** 0.5))


def simulate_strategy(
    df: pl.DataFrame,
    params: StrategyParams,
    *,
    fee_category: str | None = None,
    fee_role: Literal["taker", "maker"] = "taker",
    spread_slippage_bps: float = 0.0,
    fee_model: Any | None = None,
) -> BacktestResult:
    """
    Buy 1 share at trade price; resolve to $1 if token_won else $0.

    Tracks chronological cumulative PnL, Sharpe on ROI, max drawdown, win rate.

    When ``fee_category`` is set, subtracts per-fill FeeModel fees and optional
    spread slippage from gross PnL. Fee categories absent from lake metadata
    must be treated as conservative fallbacks and labeled separately.
    """
    filtered = apply_strategy_filter(df, params)
    edge = compute_edge_stats(filtered, params)

    if filtered.is_empty():
        return BacktestResult(
            n=0,
            win_rate=0.0,
            implied_win_rate=0.0,
            ev_pct=0.0,
            total_pnl=0.0,
            avg_pnl=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            params=asdict(params),
            label=params.label(),
            equity_curve=[],
        )

    ordered = filtered.sort("traded_at")
    if fee_category is not None or spread_slippage_bps:
        from polymarket_analytics.research.fees import FeeModel, compute_fill_fee

        fm = fee_model or FeeModel()
        slip = max(float(spread_slippage_bps), 0.0) / 10_000.0
        fees: list[float] = []
        fill_px: list[float] = []
        for row in ordered.select(["price", "traded_at"]).iter_rows(named=True):
            px = float(row["price"]) * (1.0 + slip)
            px = min(max(px, 0.01), 0.99)
            fill_px.append(px)
            if fee_category is None:
                fees.append(0.0)
            else:
                info = compute_fill_fee(
                    1.0,
                    px,
                    role=fee_role,
                    category=fee_category,
                    as_of=row["traded_at"],
                    model=fm,
                )
                fees.append(float(info["fee"]))
        ordered = ordered.with_columns(
            pl.Series("fill_price", fill_px),
            pl.Series("fee", fees),
        ).with_columns(
            (
                pl.col("token_won").cast(pl.Float64)
                - pl.col("fill_price")
                - pl.col("fee")
            ).alias("pnl"),
            (
                (
                    pl.col("token_won").cast(pl.Float64)
                    - pl.col("fill_price")
                    - pl.col("fee")
                )
                / pl.col("fill_price").clip(lower_bound=1e-12)
            ).alias("roi"),
        )
        # Net EV% uses fee-aware fill prices
        net_ev = 100.0 * float(ordered["pnl"].mean())
        edge = EdgeStats(
            n=edge.n,
            empirical_win_rate=edge.empirical_win_rate,
            implied_win_rate=float(ordered["fill_price"].mean()),
            ev_pct=net_ev,
            params=asdict(params),
            label=params.label(),
        )
    else:
        ordered = ordered.with_columns(
            (pl.col("token_won").cast(pl.Float64) - pl.col("price")).alias("pnl"),
            (
                (pl.col("token_won").cast(pl.Float64) - pl.col("price"))
                / pl.col("price").clip(lower_bound=1e-12)
            ).alias("roi"),
        )

    pnls = ordered["pnl"]
    equity = pnls.cum_sum().to_list()
    total_pnl = float(pnls.sum())
    avg_pnl = float(pnls.mean())
    sharpe = _sharpe_ratio(ordered["roi"])
    max_dd = _max_drawdown(equity)

    return BacktestResult(
        n=edge.n,
        win_rate=edge.empirical_win_rate,
        implied_win_rate=edge.implied_win_rate,
        ev_pct=edge.ev_pct,
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        sharpe=sharpe,
        max_drawdown=max_dd,
        params=asdict(params),
        label=params.label(),
        equity_curve=[float(x) for x in equity],
    )


def iter_grid_params(
    *,
    price_buckets: Sequence[str] | None = None,
    spike_thresholds: Sequence[float | None] = DEFAULT_SPIKE_THRESHOLDS,
    whale_thresholds: Sequence[float | None] = DEFAULT_WHALE_THRESHOLDS,
    ttr_bounds: Sequence[float | None] = DEFAULT_TTR_BOUNDS,
    momentum_1h_signs: Sequence[MomentumSign] = DEFAULT_MOMENTUM_SIGNS,
    momentum_6h_signs: Sequence[MomentumSign] = ("any",),
    side: str | None = "BUY",
) -> Iterable[StrategyParams]:
    """Cartesian product of discrete filter axes for edge scanning."""
    buckets: Sequence[str | None]
    if price_buckets is None:
        buckets = list(PRICE_BUCKET_LABELS)
    else:
        buckets = list(price_buckets)

    for bucket, spike, whale, ttr, m1, m6 in product(
        buckets,
        spike_thresholds,
        whale_thresholds,
        ttr_bounds,
        momentum_1h_signs,
        momentum_6h_signs,
    ):
        yield StrategyParams(
            price_bucket=bucket,
            min_volume_spike=spike,
            min_whale_ratio=whale,
            max_time_to_resolution_hours=ttr,
            momentum_1h=m1,
            momentum_6h=m6,
            side=side,
        )


def find_edges(
    df: pl.DataFrame,
    *,
    min_samples: int = 5,
    min_ev_pct: float | None = None,
    price_buckets: Sequence[str] | None = None,
    spike_thresholds: Sequence[float | None] = DEFAULT_SPIKE_THRESHOLDS,
    whale_thresholds: Sequence[float | None] = DEFAULT_WHALE_THRESHOLDS,
    ttr_bounds: Sequence[float | None] = DEFAULT_TTR_BOUNDS,
    momentum_1h_signs: Sequence[MomentumSign] = DEFAULT_MOMENTUM_SIGNS,
    momentum_6h_signs: Sequence[MomentumSign] = ("any",),
    side: str | None = "BUY",
) -> pl.DataFrame:
    """
    Grid-search ``v_trade_features`` slices; return ranked edge table (Polars).

    Sorted by ``ev_pct`` descending, then ``n`` descending.
    """
    rows: list[dict[str, Any]] = []
    for params in iter_grid_params(
        price_buckets=price_buckets,
        spike_thresholds=spike_thresholds,
        whale_thresholds=whale_thresholds,
        ttr_bounds=ttr_bounds,
        momentum_1h_signs=momentum_1h_signs,
        momentum_6h_signs=momentum_6h_signs,
        side=side,
    ):
        sliced = apply_strategy_filter(df, params)
        if sliced.height < min_samples:
            continue
        stats = compute_edge_stats(sliced, params)
        if min_ev_pct is not None and stats.ev_pct < min_ev_pct:
            continue
        rows.append(
            {
                "label": stats.label,
                "n": stats.n,
                "empirical_win_rate": stats.empirical_win_rate,
                "implied_win_rate": stats.implied_win_rate,
                "ev_pct": stats.ev_pct,
                "price_bucket": params.price_bucket,
                "min_volume_spike": params.min_volume_spike,
                "min_whale_ratio": params.min_whale_ratio,
                "momentum_1h": params.momentum_1h,
                "momentum_6h": params.momentum_6h,
                "max_time_to_resolution_hours": params.max_time_to_resolution_hours,
                "side": params.side,
            }
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "label": pl.Utf8,
                "n": pl.Int64,
                "empirical_win_rate": pl.Float64,
                "implied_win_rate": pl.Float64,
                "ev_pct": pl.Float64,
                "price_bucket": pl.Utf8,
                "min_volume_spike": pl.Float64,
                "min_whale_ratio": pl.Float64,
                "momentum_1h": pl.Utf8,
                "momentum_6h": pl.Utf8,
                "max_time_to_resolution_hours": pl.Float64,
                "side": pl.Utf8,
            }
        )

    return pl.DataFrame(rows).sort(["ev_pct", "n"], descending=[True, True])


def parse_date_bound(value: str | date | datetime | None) -> datetime | None:
    """Parse YYYY-MM-DD (or datetime) to UTC midnight datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    text = str(value).strip()
    # Allow full ISO timestamps too
    if "T" in text or " " in text:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    y, m, d = (int(x) for x in text.split("-", 2))
    return datetime(y, m, d, tzinfo=timezone.utc)


def filter_by_traded_at(
    df: pl.DataFrame,
    *,
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
    start_inclusive: bool = True,
    end_inclusive: bool = False,
) -> pl.DataFrame:
    """
    Isolate rows by ``traded_at`` window.

    Default half-open interval [start, end) so train/test splits do not overlap.
    """
    if df.is_empty() or "traded_at" not in df.columns:
        return df

    # Normalize tz so DuckDB-local timestamps compare cleanly to UTC bounds
    traded = pl.col("traded_at")
    dtype = df.schema["traded_at"]
    if isinstance(dtype, pl.Datetime) and dtype.time_zone not in (None, "UTC"):
        traded = traded.dt.convert_time_zone("UTC")
    elif isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
        traded = traded.dt.replace_time_zone("UTC")
    traded = traded.cast(pl.Datetime("us", "UTC"))

    start_dt = parse_date_bound(start)
    end_dt = parse_date_bound(end)
    exprs: list[pl.Expr] = []
    if start_dt is not None:
        start_lit = pl.lit(start_dt).cast(pl.Datetime("us", "UTC"))
        if start_inclusive:
            exprs.append(traded >= start_lit)
        else:
            exprs.append(traded > start_lit)
    if end_dt is not None:
        end_lit = pl.lit(end_dt).cast(pl.Datetime("us", "UTC"))
        if end_inclusive:
            exprs.append(traded <= end_lit)
        else:
            exprs.append(traded < end_lit)
    if not exprs:
        return df
    mask = exprs[0]
    for e in exprs[1:]:
        mask = mask & e
    return df.filter(mask)


def params_from_edge_row(row: dict[str, Any]) -> StrategyParams:
    """Rebuild StrategyParams from a find_edges result row."""
    return StrategyParams(
        price_bucket=row.get("price_bucket"),
        min_volume_spike=row.get("min_volume_spike"),
        min_whale_ratio=row.get("min_whale_ratio"),
        momentum_1h=row.get("momentum_1h") or "any",
        momentum_6h=row.get("momentum_6h") or "any",
        max_time_to_resolution_hours=row.get("max_time_to_resolution_hours"),
        side=row.get("side"),
    )


def evaluate_setup(df: pl.DataFrame, params: StrategyParams) -> EdgeStats:
    """Apply filters and compute edge stats on a (possibly date-sliced) frame."""
    return compute_edge_stats(apply_strategy_filter(df, params), params)


def format_oos_comparison_table(rows: Sequence[dict[str, Any]]) -> str:
    """Human-readable IS vs OOS comparison table."""
    headers = (
        "label",
        "is_n",
        "is_win%",
        "is_ev",
        "oos_n",
        "oos_win%",
        "oos_ev",
        "ev_decay",
    )
    lines = [" | ".join(headers), " | ".join("-" * len(h) for h in headers)]
    for r in rows:
        lines.append(
            " | ".join(
                [
                    str(r.get("label", ""))[:48],
                    str(r.get("is_n", 0)),
                    f"{100.0 * float(r.get('is_win_rate', 0.0)):.1f}",
                    f"{float(r.get('is_ev_pct', 0.0)):.2f}",
                    str(r.get("oos_n", 0)),
                    f"{100.0 * float(r.get('oos_win_rate', 0.0)):.1f}",
                    f"{float(r.get('oos_ev_pct', 0.0)):.2f}",
                    f"{float(r.get('ev_decay_pct', 0.0)):.2f}",
                ]
            )
        )
    return "\n".join(lines)


def run_oos_edge_validation(
    df: pl.DataFrame,
    *,
    split_date: str | date | datetime,
    min_samples: int = 20,
    min_ev_pct: float | None = None,
    price_buckets: Sequence[str] | None = None,
    spike_thresholds: Sequence[float | None] = DEFAULT_SPIKE_THRESHOLDS,
    whale_thresholds: Sequence[float | None] = DEFAULT_WHALE_THRESHOLDS,
    ttr_bounds: Sequence[float | None] = DEFAULT_TTR_BOUNDS,
    momentum_1h_signs: Sequence[MomentumSign] = DEFAULT_MOMENTUM_SIGNS,
    momentum_6h_signs: Sequence[MomentumSign] = ("any",),
    side: str | None = "BUY",
    top_k: int = 20,
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
) -> dict[str, Any]:
    """
    Discover edges on the in-sample window, then re-score the same setups OOS.

    Train: traded_at ∈ [start, split_date)
    Test:  traded_at ∈ [split_date, end)
    """
    split_dt = parse_date_bound(split_date)
    assert split_dt is not None

    scoped = filter_by_traded_at(df, start=start, end=end)
    train = filter_by_traded_at(scoped, end=split_dt)  # [..., split)
    test = filter_by_traded_at(scoped, start=split_dt)  # [split, ...)

    is_edges = find_edges(
        train,
        min_samples=min_samples,
        min_ev_pct=min_ev_pct,
        price_buckets=price_buckets,
        spike_thresholds=spike_thresholds,
        whale_thresholds=whale_thresholds,
        ttr_bounds=ttr_bounds,
        momentum_1h_signs=momentum_1h_signs,
        momentum_6h_signs=momentum_6h_signs,
        side=side,
    )
    if top_k is not None and is_edges.height > top_k:
        is_edges = is_edges.head(top_k)

    comparisons: list[dict[str, Any]] = []
    for row in is_edges.to_dicts():
        params = params_from_edge_row(row)
        oos = evaluate_setup(test, params)
        is_ev = float(row["ev_pct"])
        comparisons.append(
            {
                "label": row["label"],
                "params": asdict(params),
                "is_n": int(row["n"]),
                "is_win_rate": float(row["empirical_win_rate"]),
                "is_implied_win_rate": float(row["implied_win_rate"]),
                "is_ev_pct": is_ev,
                "oos_n": oos.n,
                "oos_win_rate": oos.empirical_win_rate,
                "oos_implied_win_rate": oos.implied_win_rate,
                "oos_ev_pct": oos.ev_pct,
                "ev_decay_pct": is_ev - oos.ev_pct,
                "persists": oos.n >= min_samples and oos.ev_pct > 0,
            }
        )

    return {
        "split_date": split_dt.date().isoformat(),
        "start": parse_date_bound(start).date().isoformat() if start else None,
        "end": parse_date_bound(end).date().isoformat() if end else None,
        "min_samples": min_samples,
        "train_rows": train.height,
        "test_rows": test.height,
        "n_is_edges": is_edges.height,
        "comparisons": comparisons,
        "is_edges": is_edges.to_dicts(),
    }


def run_find_edges(
    warehouse_path: Path | str,
    parquet_dir: Path | str | None = None,
    *,
    min_samples: int = 5,
    min_ev_pct: float | None = None,
    price_buckets: Sequence[str] | None = None,
    min_whale_ratio: float | None = None,
    max_ttr_hours: float | None = None,
    top_k: int | None = 50,
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
    split_date: str | date | datetime | None = None,
    oos_report_path: Path | str | None = None,
) -> pl.DataFrame | dict[str, Any]:
    """
    Load features and run the edge grid scan.

    If ``split_date`` is set, returns an OOS validation report dict (and optionally
    writes ``oos_report_path``). Otherwise returns a ranked edges DataFrame.
    """
    df = load_trade_features(warehouse_path, parquet_dir)
    df = filter_by_traded_at(df, start=start, end=end)

    whale_thresholds: Sequence[float | None]
    if min_whale_ratio is not None:
        whale_thresholds = (min_whale_ratio,)
    else:
        whale_thresholds = DEFAULT_WHALE_THRESHOLDS

    ttr_bounds: Sequence[float | None]
    if max_ttr_hours is not None:
        ttr_bounds = (max_ttr_hours,)
    else:
        ttr_bounds = DEFAULT_TTR_BOUNDS

    if split_date is not None:
        report = run_oos_edge_validation(
            df,
            split_date=split_date,
            min_samples=min_samples,
            min_ev_pct=min_ev_pct,
            price_buckets=price_buckets,
            whale_thresholds=whale_thresholds,
            ttr_bounds=ttr_bounds,
            top_k=top_k or 20,
            start=None,  # already applied above
            end=None,
        )
        if oos_report_path is not None:
            path = Path(oos_report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            import json

            path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            report["report_path"] = str(path)
        return report

    edges = find_edges(
        df,
        min_samples=min_samples,
        min_ev_pct=min_ev_pct,
        price_buckets=price_buckets,
        whale_thresholds=whale_thresholds,
        ttr_bounds=ttr_bounds,
    )
    if top_k is not None and edges.height > top_k:
        edges = edges.head(top_k)
    return edges


def run_backtest(
    warehouse_path: Path | str,
    params: StrategyParams,
    parquet_dir: Path | str | None = None,
    *,
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
) -> BacktestResult:
    """Load features and simulate one strategy parameter set."""
    df = load_trade_features(warehouse_path, parquet_dir)
    df = filter_by_traded_at(df, start=start, end=end)
    return simulate_strategy(df, params)
