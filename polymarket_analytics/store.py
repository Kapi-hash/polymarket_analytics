"""DuckDB warehouse: register Parquet lake and analytics views."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import duckdb
import polars as pl

from polymarket_analytics.schema import (
    EMPTY_TRADES_SEED_DATE,
    FEATURES_SCHEMA,
    MARKET_TOKENS_SCHEMA,
    MARKETS_SCHEMA,
    TRADES_SCHEMA,
    duckdb_ddl,
    parquet_view_sql,
)


class WarehouseStatus(TypedDict):
    warehouse: str
    trades: int
    markets: int
    resolved_markets: int
    resolved_market_pct: float
    traded_at_min: Any
    traded_at_max: Any
    resolved_trades: int
    feature_rows: int


def connect(warehouse_path: Path | str) -> duckdb.DuckDBPyConnection:
    path = Path(warehouse_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def _as_glob(path: Path) -> str:
    return str(path).replace("\\", "/")


def _ensure_empty_lake(parquet_dir: Path) -> None:
    """Create empty Parquet placeholders so DuckDB views always resolve."""
    markets_path = parquet_dir / "markets" / "markets.parquet"
    tokens_path = parquet_dir / "market_tokens" / "market_tokens.parquet"
    features_path = parquet_dir / "features" / "trade_features.parquet"
    trades_root = parquet_dir / "trades"

    if not markets_path.exists():
        markets_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(schema=MARKETS_SCHEMA).write_parquet(markets_path)

    if not tokens_path.exists():
        tokens_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(schema=MARKET_TOKENS_SCHEMA).write_parquet(tokens_path)

    if not features_path.exists():
        features_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(schema=FEATURES_SCHEMA).write_parquet(features_path)

    trades_any = list(trades_root.rglob("*.parquet")) if trades_root.exists() else []
    if not trades_any:
        seed_dir = trades_root / f"date={EMPTY_TRADES_SEED_DATE}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(schema=TRADES_SCHEMA).write_parquet(seed_dir / "part-000.parquet")


def bootstrap_warehouse(parquet_dir: Path | str, warehouse_path: Path | str) -> Path:
    """Create/refresh DuckDB tables + views over the Parquet lake."""
    parquet_dir = Path(parquet_dir)
    warehouse_path = Path(warehouse_path)
    _ensure_empty_lake(parquet_dir)

    trades_glob = _as_glob(parquet_dir / "trades" / "**" / "*.parquet")
    markets_glob = _as_glob(parquet_dir / "markets" / "*.parquet")
    tokens_glob = _as_glob(parquet_dir / "market_tokens" / "*.parquet")
    features_glob = _as_glob(parquet_dir / "features" / "*.parquet")

    con = connect(warehouse_path)
    try:
        for stmt in duckdb_ddl().split(";"):
            stmt = stmt.strip()
            if stmt:
                con.execute(stmt)
        con.execute(
            parquet_view_sql(
                trades_glob,
                markets_glob,
                tokens_glob,
                features_glob=features_glob,
            )
        )
    finally:
        con.close()
    return warehouse_path


def warehouse_status(
    warehouse_path: Path | str,
    parquet_dir: Path | str | None = None,
) -> WarehouseStatus:
    """Cheap aggregates for CLI status (pushdown via DuckDB over Parquet)."""
    warehouse_path = Path(warehouse_path)
    if parquet_dir is not None:
        bootstrap_warehouse(parquet_dir, warehouse_path)

    con = connect(warehouse_path)
    try:
        trade_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM v_trades
            WHERE traded_at > TIMESTAMP '{EMPTY_TRADES_SEED_DATE}'
            """
        ).fetchone()[0]
        market_count = con.execute("SELECT COUNT(*) FROM v_markets").fetchone()[0]
        resolved_count = con.execute(
            "SELECT COUNT(*) FROM v_markets WHERE resolved = TRUE"
        ).fetchone()[0]
        date_range = con.execute(
            f"""
            SELECT MIN(traded_at), MAX(traded_at)
            FROM v_trades
            WHERE traded_at > TIMESTAMP '{EMPTY_TRADES_SEED_DATE}'
            """
        ).fetchone()
        resolved_trades = con.execute("SELECT COUNT(*) FROM v_resolved_trades").fetchone()[0]
        try:
            feature_rows = con.execute("SELECT COUNT(*) FROM v_features").fetchone()[0]
        except duckdb.Error:
            feature_rows = 0
        return {
            "warehouse": str(warehouse_path),
            "trades": int(trade_count),
            "markets": int(market_count),
            "resolved_markets": int(resolved_count),
            "resolved_market_pct": (
                round(100.0 * resolved_count / market_count, 2) if market_count else 0.0
            ),
            "traded_at_min": date_range[0],
            "traded_at_max": date_range[1],
            "resolved_trades": int(resolved_trades),
            "feature_rows": int(feature_rows),
        }
    finally:
        con.close()
