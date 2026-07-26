"""Polars-vectorized, chunked ingest into a partitioned Parquet lake."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import polars as pl

from polymarket_analytics.schema import (
    MARKET_ALIASES,
    MARKET_TOKENS_COLUMNS,
    MARKET_TOKENS_SCHEMA,
    MARKETS_COLUMNS,
    MARKETS_SCHEMA,
    TRADE_ALIASES,
    TRADES_COLUMNS,
    TRADES_SCHEMA,
)

SUPPORTED_SUFFIXES: frozenset[str] = frozenset(
    {".csv", ".json", ".ndjson", ".parquet", ".jsonl"}
)

# Default chunk size for CSV / NDJSON streaming (rows).
DEFAULT_CHUNK_ROWS: int = 500_000


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _list_raw_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(
        p
        for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _rename_aliases(df: pl.DataFrame, aliases: dict[str, str]) -> pl.DataFrame:
    rename_map: dict[str, str] = {}
    claimed: set[str] = set()
    for col in df.columns:
        target = aliases.get(col)
        if target is None:
            continue
        if target in claimed:
            continue
        if target in df.columns and col != target:
            # Prefer the already-canonical column name.
            claimed.add(target)
            continue
        rename_map[col] = target
        claimed.add(target)
    return df.rename(rename_map) if rename_map else df


def _ensure_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if not missing:
        return df
    return df.with_columns([pl.lit(None).alias(c) for c in missing])


def _parse_utc_datetime(expr: pl.Expr) -> pl.Expr:
    """Epoch s/ms/µs or ISO-8601 string → Datetime(us, UTC).

    ISO parsing uses explicit formats with ``strict=False`` so pure epoch-digit
    chunks never raise ComputeError. Native Datetime columns should be
    normalized via ``_coerce_datetime_column`` before this is applied.
    """
    as_str = expr.cast(pl.Utf8, strict=False)
    as_num = as_str.cast(pl.Float64, strict=False)
    seconds = (
        pl.when(as_num.abs() > 1e14)
        .then(as_num / 1_000_000.0)
        .when(as_num.abs() > 1e11)
        .then(as_num / 1000.0)
        .otherwise(as_num)
    )
    from_epoch = pl.from_epoch(seconds.cast(pl.Int64), time_unit="s").dt.replace_time_zone(
        "UTC"
    )
    iso_z = as_str.str.to_datetime(
        format="%Y-%m-%dT%H:%M:%SZ",
        time_unit="us",
        time_zone="UTC",
        strict=False,
    )
    iso_frac_z = as_str.str.to_datetime(
        format="%Y-%m-%dT%H:%M:%S%.fZ",
        time_unit="us",
        time_zone="UTC",
        strict=False,
    )
    iso_offset = as_str.str.to_datetime(
        format="%Y-%m-%dT%H:%M:%S%z",
        time_unit="us",
        time_zone="UTC",
        strict=False,
    )
    return pl.coalesce(from_epoch, iso_z, iso_frac_z, iso_offset)


def _coerce_datetime_column(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Normalize a column that may already be Datetime or still be string/epoch."""
    if col not in df.columns:
        return df
    dtype = df.schema[col]
    if isinstance(dtype, pl.Datetime):
        series = df[col]
        if dtype.time_zone is None:
            series = series.dt.replace_time_zone("UTC")
        else:
            series = series.dt.convert_time_zone("UTC")
        return df.with_columns(series.cast(pl.Datetime("us", "UTC")).alias(col))
    return df.with_columns(_parse_utc_datetime(pl.col(col)).alias(col))


def _trade_id_expr() -> pl.Expr:
    """Stable SHA1 over identity fields (fully vectorized via Polars hash → hex)."""
    key = pl.concat_str(
        [
            pl.col("tx_hash").fill_null(""),
            pl.lit("|"),
            pl.col("token_id").fill_null(""),
            pl.lit("|"),
            pl.col("traded_at").cast(pl.Utf8).fill_null(""),
            pl.lit("|"),
            pl.col("price").cast(pl.Utf8).fill_null(""),
            pl.lit("|"),
            pl.col("size").cast(pl.Utf8).fill_null(""),
            pl.lit("|"),
            pl.col("side").cast(pl.Utf8).fill_null(""),
        ]
    )
    # hash() is uint64; combine with a second salt hash for a 128-bit-ish hex id.
    return pl.concat_str(
        [
            key.hash(seed=0).cast(pl.Utf8),
            pl.lit("_"),
            key.hash(seed=1).cast(pl.Utf8),
        ]
    ).alias("trade_id")


def normalize_trades(df: pl.DataFrame, *, source_file: str) -> pl.DataFrame:
    """Vectorized normalize / validate / dedupe-prep for one trade chunk."""
    if df.is_empty():
        return pl.DataFrame(schema=TRADES_SCHEMA)

    df = _rename_aliases(df, TRADE_ALIASES)
    df = _ensure_columns(
        df,
        [
            "trade_id",
            "condition_id",
            "token_id",
            "side",
            "price",
            "size",
            "outcome_label",
            "outcome_index",
            "wallet",
            "tx_hash",
            "traded_at",
        ],
    )

    df = df.with_columns(
        [
            pl.col("condition_id").cast(pl.Utf8, strict=False),
            pl.col("token_id").cast(pl.Utf8, strict=False),
            pl.col("side").cast(pl.Utf8, strict=False).str.to_uppercase(),
            pl.col("price").cast(pl.Float64, strict=False),
            pl.col("size").cast(pl.Float64, strict=False),
            pl.col("outcome_label").cast(pl.Utf8, strict=False),
            pl.col("outcome_index").cast(pl.Int8, strict=False),
            pl.col("wallet").cast(pl.Utf8, strict=False),
            pl.col("tx_hash").cast(pl.Utf8, strict=False),
            pl.col("trade_id").cast(pl.Utf8, strict=False),
        ]
    )
    df = _coerce_datetime_column(df, "traded_at")


    df = df.filter(
        pl.col("condition_id").is_not_null()
        & (pl.col("condition_id") != "")
        & pl.col("token_id").is_not_null()
        & (pl.col("token_id") != "")
        & pl.col("price").is_not_null()
        & pl.col("size").is_not_null()
        & pl.col("traded_at").is_not_null()
        & (pl.col("price") >= 0.0)
        & (pl.col("price") <= 1.0)
        & (pl.col("size") > 0.0)
        & pl.col("side").is_in(["BUY", "SELL"])
    )

    if df.is_empty():
        return pl.DataFrame(schema=TRADES_SCHEMA)

    needs_id = pl.col("trade_id").is_null() | (pl.col("trade_id") == "")
    df = df.with_columns(
        pl.when(needs_id).then(_trade_id_expr()).otherwise(pl.col("trade_id")).alias("trade_id")
    )

    ingest_at = _utc_now()
    df = df.with_columns(
        [
            (pl.col("price") * pl.col("size")).alias("notional"),
            pl.lit(ingest_at).cast(pl.Datetime("us", "UTC")).alias("ingest_at"),
            pl.lit(source_file).alias("source_file"),
            pl.col("side").cast(pl.Categorical),
        ]
    )
    return df.select(TRADES_COLUMNS).cast(TRADES_SCHEMA)


def _explode_market_tokens(markets: pl.DataFrame) -> pl.DataFrame:
    """Vectorized Yes/No token expansion (no Python row loop)."""
    if markets.is_empty():
        return pl.DataFrame(schema=MARKET_TOKENS_SCHEMA)

    yes = markets.select(
        pl.col("condition_id"),
        pl.col("token_yes").alias("token_id"),
        pl.lit("Yes").alias("outcome_label"),
        pl.lit(0, dtype=pl.Int8).alias("outcome_index"),
    ).filter(pl.col("token_id").is_not_null() & (pl.col("token_id") != ""))

    no = markets.select(
        pl.col("condition_id"),
        pl.col("token_no").alias("token_id"),
        pl.lit("No").alias("outcome_label"),
        pl.lit(1, dtype=pl.Int8).alias("outcome_index"),
    ).filter(pl.col("token_id").is_not_null() & (pl.col("token_id") != ""))

    if yes.is_empty() and no.is_empty():
        return pl.DataFrame(schema=MARKET_TOKENS_SCHEMA)
    return (
        pl.concat([yes, no], how="vertical_relaxed")
        .cast(MARKET_TOKENS_SCHEMA)
        .select(MARKET_TOKENS_COLUMNS)
    )


def _parse_json_list(s: object) -> list[str] | None:
    if not isinstance(s, str) or not s.startswith("["):
        return None
    try:
        value = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list):
        return None
    return [str(x) for x in value]


def _winner_label_from_prices(s: object) -> str | None:
    if s is None:
        return None
    try:
        prices = json.loads(s) if isinstance(s, str) else s
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(prices, list) or not prices:
        return None
    try:
        floats = [float(p) for p in prices]
    except (TypeError, ValueError):
        return None
    if max(floats) < 0.99:
        return None
    idx = floats.index(max(floats))
    if idx == 0:
        return "Yes"
    if idx == 1:
        return "No"
    return str(idx)


def normalize_markets(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Normalize market outcome metadata + derive market_tokens."""
    if df.is_empty():
        empty_m = pl.DataFrame(schema=MARKETS_SCHEMA)
        empty_t = pl.DataFrame(schema=MARKET_TOKENS_SCHEMA)
        return empty_m, empty_t

    df = _rename_aliases(df, MARKET_ALIASES)

    if "clob_token_ids" in df.columns and (
        "token_yes" not in df.columns or df["token_yes"].null_count() == df.height
    ):
        parsed = df["clob_token_ids"].map_elements(
            _parse_json_list, return_dtype=pl.List(pl.Utf8)
        )
        df = df.with_columns(
            [
                parsed.list.get(0).alias("token_yes"),
                parsed.list.get(1).alias("token_no"),
            ]
        )

    if "outcome_prices" in df.columns:
        winners = df["outcome_prices"].map_elements(
            _winner_label_from_prices, return_dtype=pl.Utf8
        )
        if "winning_outcome" not in df.columns:
            df = df.with_columns(winners.alias("winning_outcome"))
        else:
            df = df.with_columns(
                pl.coalesce(
                    pl.col("winning_outcome").cast(pl.Utf8, strict=False), winners
                ).alias("winning_outcome")
            )

    df = _ensure_columns(
        df,
        [
            "condition_id",
            "question",
            "slug",
            "token_yes",
            "token_no",
            "winning_token_id",
            "winning_outcome",
            "resolved",
            "resolved_at",
            "closed_at",
            "end_date",
            "volume",
            "liquidity",
            "neg_risk",
            "closed",
        ],
    )

    df = df.with_columns(
        [
            pl.col("condition_id").cast(pl.Utf8, strict=False),
            pl.col("question").cast(pl.Utf8, strict=False),
            pl.col("slug").cast(pl.Utf8, strict=False),
            pl.col("token_yes").cast(pl.Utf8, strict=False),
            pl.col("token_no").cast(pl.Utf8, strict=False),
            pl.col("winning_token_id").cast(pl.Utf8, strict=False),
            pl.col("winning_outcome").cast(pl.Utf8, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False),
            pl.col("liquidity").cast(pl.Float64, strict=False),
            pl.col("neg_risk").cast(pl.Boolean, strict=False).fill_null(False),
        ]
    )
    for dt_col in ("resolved_at", "closed_at", "end_date"):
        df = _coerce_datetime_column(df, dt_col)


    df = df.with_columns(
        pl.when(pl.col("winning_token_id").is_not_null() & (pl.col("winning_token_id") != ""))
        .then(pl.col("winning_token_id"))
        .when(pl.col("winning_outcome").str.to_lowercase().is_in(["yes", "y"]))
        .then(pl.col("token_yes"))
        .when(pl.col("winning_outcome").str.to_lowercase().is_in(["no", "n"]))
        .then(pl.col("token_no"))
        .otherwise(None)
        .alias("winning_token_id")
    )

    closed_bool = (
        pl.col("closed")
        .cast(pl.Utf8, strict=False)
        .str.to_lowercase()
        .is_in(["true", "1", "yes"])
    )
    df = df.with_columns(
        pl.coalesce(
            pl.col("resolved").cast(pl.Boolean, strict=False),
            pl.col("winning_token_id").is_not_null(),
            closed_bool,
            pl.lit(False),
        ).alias("resolved")
    )
    df = df.with_columns(
        pl.lit(_utc_now()).cast(pl.Datetime("us", "UTC")).alias("ingest_at")
    )
    df = df.filter(pl.col("condition_id").is_not_null() & (pl.col("condition_id") != ""))

    markets = (
        df.select(MARKETS_COLUMNS).cast(MARKETS_SCHEMA).unique(subset=["condition_id"], keep="last")
    )
    return markets, _explode_market_tokens(markets)


# ---------------------------------------------------------------------------
# Chunked readers (bounded peak RAM)
# ---------------------------------------------------------------------------


def iter_raw_chunks(
    path: Path, *, chunk_rows: int = DEFAULT_CHUNK_ROWS
) -> Iterator[pl.DataFrame]:
    """Yield DataFrame chunks from a raw file without loading the whole file when possible."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        reader = pl.read_csv_batched(
            path,
            batch_size=chunk_rows,
            infer_schema_length=10_000,
            ignore_errors=True,
        )
        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            yield batches[0]
        return

    if suffix == ".parquet":
        # Slice by row groups via lazy scan + streaming collect slices.
        lf = pl.scan_parquet(path)
        n = lf.select(pl.len()).collect().item()
        if n == 0:
            yield pl.DataFrame()
            return
        for start in range(0, int(n), chunk_rows):
            yield lf.slice(start, chunk_rows).collect()
        return

    if suffix in {".ndjson", ".jsonl"}:
        # Polars has no NDJSON batched reader; stream via offset slices on lazy scan.
        lf = pl.scan_ndjson(path)
        # Materialize in chunks via collecting with slice after reading in one pass
        # is still heavy for huge files — fall back to full read for NDJSON.
        # Prefer converting dumps to Parquet/CSV for multi-GB feeds.
        df = pl.read_ndjson(path)
        if df.height <= chunk_rows:
            yield df
            return
        for start in range(0, df.height, chunk_rows):
            yield df.slice(start, chunk_rows)
        return

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], list):
                payload = payload["data"]
            else:
                payload = [payload]
        if not isinstance(payload, list):
            raise ValueError(f"Unsupported JSON root in {path}")
        df = pl.DataFrame(payload, infer_schema_length=10_000)
        if df.height <= chunk_rows:
            yield df
            return
        for start in range(0, df.height, chunk_rows):
            yield df.slice(start, chunk_rows)
        return

    raise ValueError(f"Unsupported file type: {path}")


# ---------------------------------------------------------------------------
# Parquet writers
# ---------------------------------------------------------------------------


def write_trades_parquet(df: pl.DataFrame, out_dir: Path) -> int:
    """Write/merge trades into hive partitions: trades/date=YYYY-MM-DD/part-000.parquet."""
    if df.is_empty():
        return 0

    trades_dir = out_dir / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    with_date = df.with_columns(pl.col("traded_at").dt.strftime("%Y-%m-%d").alias("date"))
    total = 0

    for date in with_date["date"].unique().sort().to_list():
        part = with_date.filter(pl.col("date") == date).drop("date")
        part_dir = trades_dir / f"date={date}"
        part_dir.mkdir(parents=True, exist_ok=True)
        existing_files = list(part_dir.glob("*.parquet"))
        if existing_files:
            existing = pl.concat(
                [pl.read_parquet(f) for f in existing_files], how="diagonal_relaxed"
            )
            part = pl.concat([existing, part], how="diagonal_relaxed").unique(
                subset=["trade_id"], keep="last"
            )
            for f in existing_files:
                f.unlink(missing_ok=True)
        part.write_parquet(part_dir / "part-000.parquet", compression="snappy")
        total += part.height
    return total


def write_markets_parquet(
    markets: pl.DataFrame,
    tokens: pl.DataFrame,
    out_dir: Path,
) -> tuple[int, int]:
    markets_dir = out_dir / "markets"
    tokens_dir = out_dir / "market_tokens"
    markets_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir.mkdir(parents=True, exist_ok=True)

    markets_path = markets_dir / "markets.parquet"
    tokens_path = tokens_dir / "market_tokens.parquet"

    if markets_path.exists() and not markets.is_empty():
        markets = pl.concat(
            [pl.read_parquet(markets_path), markets], how="diagonal_relaxed"
        ).unique(subset=["condition_id"], keep="last")
    if tokens_path.exists() and not tokens.is_empty():
        tokens = pl.concat(
            [pl.read_parquet(tokens_path), tokens], how="diagonal_relaxed"
        ).unique(subset=["condition_id", "token_id"], keep="last")

    if not markets.is_empty():
        markets.write_parquet(markets_path, compression="snappy")
    if not tokens.is_empty():
        tokens.write_parquet(tokens_path, compression="snappy")
    return markets.height, tokens.height


def ingest_trades_file(
    path: Path, out_dir: Path, *, chunk_rows: int = DEFAULT_CHUNK_ROWS
) -> int:
    """Stream a trade file in chunks → normalize → partition-write immediately."""
    written = 0
    for chunk in iter_raw_chunks(path, chunk_rows=chunk_rows):
        normalized = normalize_trades(chunk, source_file=str(path))
        written += write_trades_parquet(normalized, out_dir)
    return written


def ingest_markets_file(path: Path, out_dir: Path) -> tuple[int, int]:
    """Markets are small; load, normalize, merge-write."""
    frames = list(iter_raw_chunks(path, chunk_rows=DEFAULT_CHUNK_ROWS))
    if not frames:
        return 0, 0
    raw = pl.concat(frames, how="diagonal_relaxed") if len(frames) > 1 else frames[0]
    markets, tokens = normalize_markets(raw)
    return write_markets_parquet(markets, tokens, out_dir)


def count_trades_lake(out_dir: Path) -> int:
    trades_dir = out_dir / "trades"
    if not trades_dir.exists():
        return 0
    files = [p for p in trades_dir.rglob("*.parquet") if "date=1970-01-01" not in str(p)]
    if not files:
        return 0
    return int(
        pl.scan_parquet([str(f) for f in files])
        .select(pl.col("trade_id").n_unique())
        .collect()
        .item()
    )


def count_table_rows(path: Path, key: str) -> int:
    if not path.exists():
        return 0
    return int(pl.scan_parquet(path).select(pl.col(key).n_unique()).collect().item())


def run_ingest(
    raw_dir: Path,
    out_dir: Path,
    *,
    bootstrap_warehouse: bool = True,
    warehouse_path: Path | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict[str, int]:
    """
    Ingest raw trade/market files into the Parquet lake.

    Trades are processed file-by-file and chunk-by-chunk so peak RAM stays near
    ``chunk_rows`` rather than the full historical universe.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in _list_raw_files(raw_dir / "trades"):
        ingest_trades_file(path, out_dir, chunk_rows=chunk_rows)

    for path in _list_raw_files(raw_dir / "markets"):
        ingest_markets_file(path, out_dir)

    trades_n = count_trades_lake(out_dir)
    markets_n = count_table_rows(out_dir / "markets" / "markets.parquet", "condition_id")
    tokens_n = count_table_rows(
        out_dir / "market_tokens" / "market_tokens.parquet", "token_id"
    )

    if bootstrap_warehouse:
        from polymarket_analytics.store import bootstrap_warehouse as _bootstrap

        wh = warehouse_path or (out_dir.parent / "warehouse.duckdb")
        _bootstrap(out_dir, wh)

    return {
        "trades": trades_n,
        "markets": markets_n,
        "market_tokens": tokens_n,
    }
