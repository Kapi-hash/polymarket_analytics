"""Phase 2 / 2.5: vectorized feature engineering (rolling, buckets, composites)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import polars as pl

from polymarket_analytics.schema import (
    DECAY_TTR_FLOOR,
    EMPTY_TRADES_SEED_DATE,
    FEATURES_COLUMNS,
    FEATURES_SCHEMA,
    FEATURE_WINDOWS,
    PRICE_BUCKET_BREAKS,
    PRICE_BUCKET_LABELS,
    WHALE_RATIO_DIVERGENCE_THRESHOLD,
)

# Minimum elapsed hours for dP/dt to avoid div-by-zero on same-timestamp fills
_EPS_HOURS: Final[float] = 1e-6


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def assign_price_bucket(price: pl.Expr) -> pl.Expr:
    """Map price ∈ [0, 1] into non-linear probability buckets."""
    return price.cut(
        PRICE_BUCKET_BREAKS,
        labels=PRICE_BUCKET_LABELS,
        left_closed=True,
    ).cast(pl.Utf8).alias("price_bucket")


def _rolling_window_frame(trades: pl.DataFrame, period: str, suffix: str) -> pl.DataFrame:
    """
    Per-row temporal rolling aggregates aligned 1:1 with ``trades`` row order.

    ``trades`` must be sorted by ``(token_id, traded_at, trade_id)``.
    Window is (t - period, t] (closed='right').
    """
    return (
        trades.rolling(
            index_column="traded_at",
            period=period,
            group_by="token_id",
            closed="right",
        )
        .agg(
            pl.col("price").first().alias("_p0"),
            pl.col("price").last().alias("_p1"),
            pl.col("traded_at").first().alias("_t0"),
            pl.col("traded_at").last().alias("_t1"),
            pl.col("price").std().alias(f"volatility_{suffix}"),
            pl.col("notional").sum().alias(f"volume_{suffix}"),
            pl.col("size").mean().alias(f"size_mean_{suffix}"),
            pl.col("size").median().alias(f"size_median_{suffix}"),
        )
        .with_columns(
            (
                (pl.col("_p1") - pl.col("_p0"))
                / (
                    (pl.col("_t1") - pl.col("_t0"))
                    .dt.total_seconds()
                    .cast(pl.Float64)
                    / 3600.0
                ).clip(lower_bound=_EPS_HOURS)
            ).alias(f"momentum_{suffix}"),
            (pl.col("_p1") - pl.col("_p0")).alias(f"price_delta_{suffix}"),
        )
        .select(
            f"momentum_{suffix}",
            f"volatility_{suffix}",
            f"volume_{suffix}",
            f"size_mean_{suffix}",
            f"size_median_{suffix}",
            f"price_delta_{suffix}",
        )
    )


def compute_rolling_features(trades: pl.DataFrame) -> pl.DataFrame:
    """Attach 1h / 6h / 24h momentum, volatility, volume, and size features."""
    if trades.is_empty():
        return trades.select("trade_id", "token_id", "traded_at")

    # Prefer explicit size; fall back to notional/price for older lakes
    if "size" not in trades.columns:
        trades = trades.with_columns(
            (pl.col("notional") / pl.col("price").clip(lower_bound=1e-12)).alias("size")
        )

    base = trades.sort(["token_id", "traded_at", "trade_id"]).select(
        "trade_id",
        "condition_id",
        "token_id",
        "traded_at",
        "price",
        "notional",
        "size",
    )

    pieces: list[pl.DataFrame] = [base]
    for period, suffix in FEATURE_WINDOWS:
        pieces.append(_rolling_window_frame(base, period, suffix))
    out = pl.concat(pieces, how="horizontal_extend")

    return out.with_columns(
        pl.when(pl.col("volume_24h").is_not_null() & (pl.col("volume_24h") > 0))
        .then(24.0 * pl.col("volume_1h") / pl.col("volume_24h"))
        .otherwise(None)
        .alias("volume_spike_1h_24h"),
        # Whale ratio: 1h mean size / 24h median size
        pl.when(
            pl.col("size_median_24h").is_not_null() & (pl.col("size_median_24h") > 0)
        )
        .then(pl.col("size_mean_1h") / pl.col("size_median_24h"))
        .otherwise(None)
        .alias("whale_ratio"),
    )


def compute_time_to_resolution(
    trades: pl.DataFrame, markets: pl.DataFrame
) -> pl.DataFrame:
    """Hours until market close/resolution: close_time - traded_at."""
    if trades.is_empty():
        return trades.select("trade_id").with_columns(
            pl.lit(None, dtype=pl.Float64).alias("time_to_resolution_hours")
        )

    close = markets.select(
        "condition_id",
        pl.coalesce(
            pl.col("resolved_at"),
            pl.col("closed_at"),
            pl.col("end_date"),
        ).alias("close_time"),
    )
    joined = trades.select("trade_id", "condition_id", "traded_at").join(
        close, on="condition_id", how="left"
    )
    return joined.select(
        "trade_id",
        (
            (pl.col("close_time") - pl.col("traded_at")).dt.total_seconds().cast(pl.Float64)
            / 3600.0
        ).alias("time_to_resolution_hours"),
    )


def compute_composite_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Phase 2.5 composites (vectorized):

    - decay_adjusted_velocity = (price_t - price_{t-1h}) / sqrt(TTR + 0.1)
    - price_volume_divergence = (price_delta_1h < 0) AND (whale_ratio > 3)
    """
    ttr = pl.col("time_to_resolution_hours")
    # Use abs(TTR) floor so negative/null TTR still yields a defined scale when possible
    denom = (ttr.fill_null(0.0).clip(lower_bound=0.0) + DECAY_TTR_FLOOR).sqrt()
    price_delta = pl.col("price_delta_1h")

    return df.with_columns(
        pl.when(price_delta.is_not_null())
        .then(price_delta / denom)
        .otherwise(None)
        .alias("decay_adjusted_velocity"),
        (
            price_delta.is_not_null()
            & (price_delta < 0)
            & pl.col("whale_ratio").is_not_null()
            & (pl.col("whale_ratio") > WHALE_RATIO_DIVERGENCE_THRESHOLD)
        ).alias("price_volume_divergence"),
    )


def compute_trade_features(
    trades: pl.DataFrame,
    markets: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Full Phase 2 / 2.5 feature matrix keyed by trade_id (vectorized Polars).

    Rolling metrics are computed per ``token_id`` over 1h / 6h / 24h windows.
    """
    if trades.is_empty():
        return pl.DataFrame(schema=FEATURES_SCHEMA)

    trades = trades.filter(pl.col("traded_at") > pl.datetime(1970, 1, 2, time_zone="UTC"))
    if trades.is_empty():
        return pl.DataFrame(schema=FEATURES_SCHEMA)

    rolling = compute_rolling_features(trades)
    rolling = rolling.with_columns(assign_price_bucket(pl.col("price")))

    if markets is not None and not markets.is_empty():
        ttr = compute_time_to_resolution(trades, markets)
        rolling = rolling.join(ttr, on="trade_id", how="left")
    else:
        rolling = rolling.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("time_to_resolution_hours")
        )

    rolling = compute_composite_features(rolling)

    feature_at = _utc_now()
    rolling = rolling.with_columns(
        pl.lit(feature_at).cast(pl.Datetime("us", "UTC")).alias("feature_at")
    )
    out = rolling.select(FEATURES_COLUMNS).cast(FEATURES_SCHEMA)
    return out.unique(subset=["trade_id"], keep="last")


def load_trades_lake(parquet_dir: Path) -> pl.DataFrame:
    trades_dir = parquet_dir / "trades"
    files = [
        p
        for p in trades_dir.rglob("*.parquet")
        if f"date={EMPTY_TRADES_SEED_DATE}" not in str(p)
    ]
    empty_schema = {
        "trade_id": pl.Utf8,
        "condition_id": pl.Utf8,
        "token_id": pl.Utf8,
        "traded_at": pl.Datetime("us", "UTC"),
        "price": pl.Float64,
        "size": pl.Float64,
        "notional": pl.Float64,
    }
    if not files:
        return pl.DataFrame(schema=empty_schema)

    lf = pl.scan_parquet([str(f) for f in files])
    cols = set(lf.collect_schema().names())
    select_cols = ["trade_id", "condition_id", "token_id", "traded_at", "price", "notional"]
    if "size" in cols:
        select_cols.append("size")
    df = lf.select(select_cols).unique(subset=["trade_id"], keep="last").collect()
    if "size" not in df.columns:
        df = df.with_columns(
            (pl.col("notional") / pl.col("price").clip(lower_bound=1e-12)).alias("size")
        )
    return df


def load_markets_lake(parquet_dir: Path) -> pl.DataFrame:
    path = parquet_dir / "markets" / "markets.parquet"
    if not path.exists():
        return pl.DataFrame(
            schema={
                "condition_id": pl.Utf8,
                "resolved_at": pl.Datetime("us", "UTC"),
                "closed_at": pl.Datetime("us", "UTC"),
                "end_date": pl.Datetime("us", "UTC"),
            }
        )
    return pl.read_parquet(path).select(
        "condition_id", "resolved_at", "closed_at", "end_date"
    )


def write_features_parquet(features: pl.DataFrame, out_dir: Path) -> Path:
    features_dir = out_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    path = features_dir / "trade_features.parquet"
    if features.is_empty():
        pl.DataFrame(schema=FEATURES_SCHEMA).write_parquet(path, compression="snappy")
    else:
        features.write_parquet(path, compression="snappy")
    return path


def run_compute_features(
    parquet_dir: Path,
    *,
    warehouse_path: Path | None = None,
    bootstrap_warehouse: bool = True,
) -> dict[str, int | str]:
    """Load lake → compute features → write Parquet → refresh DuckDB views."""
    trades = load_trades_lake(parquet_dir)
    markets = load_markets_lake(parquet_dir)
    features = compute_trade_features(trades, markets)
    path = write_features_parquet(features, parquet_dir)

    if bootstrap_warehouse:
        from polymarket_analytics.store import bootstrap_warehouse as _bootstrap

        wh = warehouse_path or (parquet_dir.parent / "warehouse.duckdb")
        _bootstrap(parquet_dir, wh)

    return {
        "rows": features.height,
        "path": str(path),
    }
