"""Duplicate trade audit and canonicalization."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import polars as pl

# Column aliases for fill identity (first match wins per slot).
_TX_COLS: tuple[str, ...] = ("tx_hash", "transaction_hash")
_LOG_COLS: tuple[str, ...] = ("log_index",)
_TOKEN_COLS: tuple[str, ...] = ("token_id", "asset_id")
_SIZE_COLS: tuple[str, ...] = ("size", "token_amount")
_TIME_COLS: tuple[str, ...] = ("traded_at", "timestamp")
_SIDE_COLS: tuple[str, ...] = ("side", "maker_direction", "taker_direction")


def _first_col(df: pl.DataFrame, candidates: Sequence[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fill_identity_columns(df: pl.DataFrame) -> list[str]:
    """Resolve available columns for fill identity key."""
    cols: list[str] = []
    for group in (_TX_COLS, _LOG_COLS, _TOKEN_COLS, ("price",), _SIZE_COLS, _TIME_COLS, _SIDE_COLS):
        picked = _first_col(df, group)
        if picked is not None:
            cols.append(picked)
    return cols


def fill_identity_key(df: pl.DataFrame) -> pl.Expr:
    """
    Build a stable string identity from available trade columns.

    Prefers (tx_hash|transaction_hash, log_index, token_id|asset_id, price,
    size|token_amount, traded_at|timestamp, side).
    """
    cols = fill_identity_columns(df)
    if not cols:
        raise ValueError("No columns available for fill identity key")
    return pl.concat_str(
        [pl.col(c).cast(pl.Utf8).fill_null("") for c in cols],
        separator="|",
    )


def audit_duplicate_trades(df: pl.DataFrame) -> dict[str, Any]:
    """
    Classify duplicate trade_ids by fill identity.

    Returns counts for exact duplicates vs field-divergent duplicates.
    """
    if df.is_empty():
        return {
            "n_rows": 0,
            "n_unique_trade_id": 0,
            "n_duplicate_trade_ids": 0,
            "n_exact_ingestion_duplicates": 0,
            "n_divergent_duplicates": 0,
            "duplicate_trade_id_count": 0,
            "identity_columns": fill_identity_columns(df),
        }

    id_cols = fill_identity_columns(df)
    keyed = df.with_columns(fill_identity_key(df).alias("_fill_key"))

    dup_ids = (
        keyed.group_by("trade_id")
        .agg(pl.len().alias("_n"))
        .filter(pl.col("_n") > 1)
    )
    n_dup_ids = dup_ids.height
    dup_trade_ids = set(dup_ids["trade_id"].to_list()) if n_dup_ids else set()

    exact = 0
    divergent = 0
    if dup_trade_ids:
        dup_rows = keyed.filter(pl.col("trade_id").is_in(list(dup_trade_ids)))
        for tid in dup_trade_ids:
            sub = dup_rows.filter(pl.col("trade_id") == tid)
            n_keys = sub["_fill_key"].n_unique()
            if n_keys == 1:
                exact += 1
            else:
                divergent += 1

    return {
        "n_rows": df.height,
        "n_unique_trade_id": df["trade_id"].n_unique() if "trade_id" in df.columns else df.height,
        "n_duplicate_trade_ids": n_dup_ids,
        "n_exact_ingestion_duplicates": exact,
        "n_divergent_duplicates": divergent,
        "duplicate_trade_id_count": int(dup_ids["_n"].sum()) if n_dup_ids else 0,
        "identity_columns": id_cols,
    }


def canonicalize_trades(df: pl.DataFrame) -> pl.DataFrame:
    """
    Deduplicate trades keeping first row by ingest_at (or stable sort).

    Adds ``fill_id`` = sha256 of fill identity key.
    """
    if df.is_empty():
        return df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("fill_id"))

    keyed = df.with_columns(fill_identity_key(df).alias("_fill_key"))
    sort_cols: list[str] = []
    if "ingest_at" in keyed.columns:
        sort_cols.append("ingest_at")
    sort_cols.extend(["traded_at", "trade_id"] if "traded_at" in keyed.columns else ["trade_id"])

    present_sort = [c for c in sort_cols if c in keyed.columns]
    if present_sort:
        keyed = keyed.sort(present_sort)

    deduped = keyed.unique(subset=["_fill_key"], keep="first")
    fill_ids = [
        hashlib.sha256(str(k).encode()).hexdigest()[:32]
        for k in deduped["_fill_key"].to_list()
    ]
    out = deduped.with_columns(pl.Series("fill_id", fill_ids)).drop("_fill_key")
    return out


def write_duplicate_audit_parquet(
    audit: dict[str, Any],
    out_path: Path | str,
) -> Path:
    """Write audit summary as a single-row parquet."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: (str(v) if isinstance(v, list) else v) for k, v in audit.items()}
    pl.DataFrame([row]).write_parquet(out_path, compression="snappy")
    return out_path


def write_duplicate_detail_parquet(
    df: pl.DataFrame,
    out_path: Path | str,
) -> Path:
    """Write duplicate trade_id detail rows with fill keys."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.is_empty() or "trade_id" not in df.columns:
        pl.DataFrame(schema={"trade_id": pl.Utf8, "_fill_key": pl.Utf8}).write_parquet(
            out_path, compression="snappy"
        )
        return out_path

    keyed = df.with_columns(fill_identity_key(df).alias("_fill_key"))
    dup_ids = (
        keyed.group_by("trade_id")
        .agg(pl.len().alias("_n"))
        .filter(pl.col("_n") > 1)["trade_id"]
        .to_list()
    )
    detail = keyed.filter(pl.col("trade_id").is_in(dup_ids))
    detail.write_parquet(out_path, compression="snappy")
    return out_path
