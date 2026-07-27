"""Build canonical deduped trade lake with market joins and features."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import polars as pl

from polymarket_analytics.features import compute_trade_features, load_markets_lake, load_trades_lake
from polymarket_analytics.research.duplicates import (
    audit_duplicate_trades,
    canonicalize_trades,
    write_duplicate_audit_parquet,
)

SourceKind = Literal["lake", "hf_sample", "merged"]


def _winner_from_outcome_prices(prices: object) -> str | None:
    if prices is None:
        return None
    try:
        if isinstance(prices, str):
            import ast
            import json

            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                prices = ast.literal_eval(prices)
        floats = [float(x) for x in list(prices)]
    except Exception:
        return None
    if not floats or max(floats) < 0.99:
        return None
    idx = floats.index(max(floats))
    return ("Yes", "No")[idx] if idx in (0, 1) else str(idx)


def _hf_sample_path(data_root: Path) -> Path:
    return data_root / "cache" / "hf" / "trades_sample_100000.parquet"


def _normalize_hf_trades(df: pl.DataFrame) -> pl.DataFrame:
    """Map HF cache columns to lake schema with proper fill identity."""
    if df.is_empty():
        return df

    out = df
    renames: dict[str, str] = {}
    if "asset_id" in out.columns and "token_id" not in out.columns:
        renames["asset_id"] = "token_id"
    if "transaction_hash" in out.columns and "tx_hash" not in out.columns:
        renames["transaction_hash"] = "tx_hash"
    if "token_amount" in out.columns and "size" not in out.columns:
        renames["token_amount"] = "size"
    if renames:
        out = out.rename(renames)

    # Epoch seconds → UTC datetime
    if "timestamp" in out.columns and "traded_at" not in out.columns:
        out = out.with_columns(
            pl.from_epoch(pl.col("timestamp").cast(pl.Int64), time_unit="s")
            .dt.replace_time_zone("UTC")
            .alias("traded_at")
        )

    if "trade_id" not in out.columns:
        if "log_index" in out.columns:
            out = out.with_columns(
                pl.concat_str(
                    [
                        pl.col("tx_hash").fill_null(""),
                        pl.lit("_"),
                        pl.col("log_index").cast(pl.Utf8).fill_null(""),
                    ]
                ).alias("trade_id")
            )
        else:
            out = out.with_row_index("_row").with_columns(
                pl.concat_str(
                    [
                        pl.col("tx_hash").fill_null(""),
                        pl.lit("|"),
                        pl.col("token_id").fill_null(""),
                        pl.lit("|"),
                        pl.col("_row").cast(pl.Utf8),
                    ]
                ).alias("trade_id")
            ).drop("_row")

    if "notional" not in out.columns and "usd_amount" in out.columns:
        out = out.with_columns(pl.col("usd_amount").cast(pl.Float64).alias("notional"))
    elif "notional" not in out.columns:
        out = out.with_columns((pl.col("size") * pl.col("price")).alias("notional"))

    if "side" not in out.columns:
        side_src = next((c for c in ("taker_direction", "maker_direction") if c in out.columns), None)
        if side_src:
            out = out.with_columns(pl.col(side_src).cast(pl.Utf8).alias("side"))

    if "wallet" not in out.columns and "taker" in out.columns:
        out = out.with_columns(pl.col("taker").alias("wallet"))

    out = out.with_columns(pl.lit("hf_sample").alias("source_file"))
    if "ingest_at" not in out.columns:
        out = out.with_columns(
            pl.lit(datetime.now(timezone.utc)).cast(pl.Datetime("us", "UTC")).alias("ingest_at")
        )
    return out


def _enrich_markets_for_resolution(markets: pl.DataFrame) -> pl.DataFrame:
    """Derive winning_token_id / resolved from HF or lake market schemas."""
    if markets.is_empty():
        return markets
    out = markets
    # HF uses token1/token2
    if "token_yes" not in out.columns and "token1" in out.columns:
        out = out.with_columns(pl.col("token1").alias("token_yes"))
    if "token_no" not in out.columns and "token2" in out.columns:
        out = out.with_columns(pl.col("token2").alias("token_no"))

    if "winning_outcome" not in out.columns and "outcome_prices" in out.columns:
        winners = out["outcome_prices"].map_elements(
            _winner_from_outcome_prices, return_dtype=pl.Utf8
        )
        out = out.with_columns(winners.alias("winning_outcome"))

    if "winning_token_id" not in out.columns and "winning_outcome" in out.columns:
        out = out.with_columns(
            pl.when(pl.col("winning_outcome") == "Yes")
            .then(pl.col("token_yes") if "token_yes" in out.columns else pl.lit(None))
            .when(pl.col("winning_outcome") == "No")
            .then(pl.col("token_no") if "token_no" in out.columns else pl.lit(None))
            .otherwise(None)
            .alias("winning_token_id")
        )

    if "resolved" not in out.columns:
        if "winning_token_id" in out.columns:
            out = out.with_columns(pl.col("winning_token_id").is_not_null().alias("resolved"))
        else:
            out = out.with_columns(pl.lit(False).alias("resolved"))

    # Prefer end_date / closed as resolved_at proxy for TTR when missing
    if "resolved_at" not in out.columns:
        if "end_date" in out.columns:
            out = out.with_columns(pl.col("end_date").alias("resolved_at"))
        else:
            out = out.with_columns(pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("resolved_at"))

    return out


def load_source_trades(
    data_root: Path | str,
    *,
    source: SourceKind = "hf_sample",
    parquet_dir: Path | str | None = None,
) -> pl.DataFrame:
    """Load trades from lake partitions, HF sample, or both (union)."""
    data_root = Path(data_root)
    if source == "hf_sample":
        path = _hf_sample_path(data_root)
        if not path.exists():
            raise FileNotFoundError(f"HF sample not found: {path}")
        return _normalize_hf_trades(pl.read_parquet(path))

    pdir = Path(parquet_dir) if parquet_dir else data_root / "parquet"
    lake = load_trades_lake(pdir)
    # Attach side/tx from full lake partitions when available
    trades_dir = pdir / "trades"
    files = list(trades_dir.rglob("*.parquet")) if trades_dir.exists() else []
    if files:
        full = pl.scan_parquet([str(f) for f in files]).collect()
        if source == "lake":
            return full
    if source == "merged":
        hf = load_source_trades(data_root, source="hf_sample")
        if lake.is_empty():
            return hf
        # Prefer HF for fill identity; append lake rows not in HF by tx+token+time
        return pl.concat([hf, lake], how="diagonal_relaxed")
    return lake if not lake.is_empty() else full  # type: ignore[name-defined]


def load_markets_enriched(
    data_root: Path | str,
    parquet_dir: Path | str | None = None,
    *,
    condition_ids: set[str] | None = None,
    prefer_hf: bool = True,
) -> pl.DataFrame:
    """Load markets with event_id and resolution fields when present."""
    data_root = Path(data_root)
    pdir = Path(parquet_dir) if parquet_dir else data_root / "parquet"
    path = pdir / "markets" / "markets.parquet"
    hf_path = data_root / "cache" / "hf" / "markets.parquet"

    hf_m = None
    if hf_path.exists():
        needed = [
            "condition_id",
            "event_id",
            "token1",
            "token2",
            "outcome_prices",
            "end_date",
            "closed",
            "volume",
            "question",
            "slug",
            "neg_risk",
        ]
        schema = pl.scan_parquet(hf_path).collect_schema().names()
        cols = [c for c in needed if c in schema]
        lf = pl.scan_parquet(hf_path).select(cols)
        if condition_ids is not None:
            lf = lf.filter(pl.col("condition_id").is_in(list(condition_ids)))
        hf_m = lf.collect()

    lake_m = None
    if path.exists() and not prefer_hf:
        lake_m = pl.read_parquet(path)
        if condition_ids is not None:
            lake_m = lake_m.filter(pl.col("condition_id").is_in(list(condition_ids)))

    if prefer_hf and hf_m is not None and not hf_m.is_empty():
        return _enrich_markets_for_resolution(hf_m)
    if lake_m is not None and not lake_m.is_empty():
        base = lake_m
        if hf_m is not None and not hf_m.is_empty():
            join_cols = [
                c
                for c in ("event_id", "token1", "token2", "outcome_prices", "end_date", "volume")
                if c in hf_m.columns
            ]
            if join_cols:
                base = base.join(
                    hf_m.select(["condition_id"] + join_cols).unique(
                        subset=["condition_id"], keep="first"
                    ),
                    on="condition_id",
                    how="left",
                )
        return _enrich_markets_for_resolution(base)
    if hf_m is not None:
        return _enrich_markets_for_resolution(hf_m)
    return load_markets_lake(pdir)


def build_canonical_lake(
    data_root: Path | str,
    *,
    source: SourceKind = "hf_sample",
    parquet_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    quality_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Canonicalize trades, join markets, recompute features, write curated parquets.

    Writes:
      data/curated/trades_canonical.parquet
      data/curated/trade_features_canonical.parquet
      data/quality/duplicate_trade_audit.parquet
    """
    data_root = Path(data_root)
    curated = Path(out_dir) if out_dir else data_root / "curated"
    curated.mkdir(parents=True, exist_ok=True)
    quality = Path(quality_dir) if quality_dir else data_root / "quality"
    quality.mkdir(parents=True, exist_ok=True)

    raw = load_source_trades(data_root, source=source, parquet_dir=parquet_dir)
    audit_before = (
        audit_duplicate_trades(raw) if not raw.is_empty() and "trade_id" in raw.columns else {}
    )
    canonical = canonicalize_trades(raw) if not raw.is_empty() else raw
    audit_after = (
        audit_duplicate_trades(canonical)
        if not canonical.is_empty() and "trade_id" in canonical.columns
        else {}
    )
    write_duplicate_audit_parquet(
        {
            "source": source,
            "n_rows_before": audit_before.get("n_rows"),
            "n_dup_ids_before": audit_before.get("n_duplicate_trade_ids"),
            "n_exact_dups_before": audit_before.get("n_exact_ingestion_duplicates"),
            "n_divergent_before": audit_before.get("n_divergent_duplicates"),
            "n_rows_after": audit_after.get("n_rows"),
            "n_dup_ids_after": audit_after.get("n_duplicate_trade_ids"),
            "identity_columns": audit_before.get("identity_columns"),
        },
        quality / "duplicate_trade_audit.parquet",
    )

    cids = set(canonical["condition_id"].drop_nulls().to_list()) if not canonical.is_empty() else set()
    markets = load_markets_enriched(
        data_root,
        parquet_dir=parquet_dir,
        condition_ids=cids or None,
        prefer_hf=(source in {"hf_sample", "merged"}),
    )
    if not canonical.is_empty() and not markets.is_empty():
        keep = [
            c
            for c in (
                "condition_id",
                "event_id",
                "winning_token_id",
                "winning_outcome",
                "resolved",
                "resolved_at",
                "end_date",
                "closed_at",
                "volume",
                "question",
                "slug",
                "neg_risk",
                "token_yes",
                "token_no",
            )
            if c in markets.columns
        ]
        mjoin = markets.select(keep).unique(subset=["condition_id"], keep="first")
        # Drop overlapping non-key cols from trades before join
        overlap = [c for c in mjoin.columns if c != "condition_id" and c in canonical.columns]
        if overlap:
            canonical = canonical.drop(overlap)
        canonical = canonical.join(mjoin, on="condition_id", how="left")
        if "winning_token_id" in canonical.columns and "token_id" in canonical.columns:
            canonical = canonical.with_columns(
                (pl.col("token_id") == pl.col("winning_token_id")).alias("token_won")
            )

    # Keep resolved trades only for outcome research
    n_before_res = canonical.height
    if "token_won" in canonical.columns:
        resolved = canonical.filter(pl.col("token_won").is_not_null())
    else:
        resolved = canonical
    n_after_res = resolved.height

    trades_path = curated / "trades_canonical.parquet"
    resolved.write_parquet(trades_path, compression="snappy")

    # Markets slice for TTR — ensure coalesce columns exist
    m_for_feat = None
    if not markets.is_empty():
        m = markets
        if "closed_at" not in m.columns:
            m = m.with_columns(pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("closed_at"))
        if "resolved_at" not in m.columns:
            m = m.with_columns(pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("resolved_at"))
        if "end_date" not in m.columns:
            m = m.with_columns(pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("end_date"))
        # Normalize end_date timezone if present
        for c in ("resolved_at", "closed_at", "end_date"):
            if c in m.columns:
                dtype = m.schema[c]
                if hasattr(dtype, "time_zone") and dtype.time_zone is None:
                    m = m.with_columns(pl.col(c).dt.replace_time_zone("UTC"))
                elif str(dtype).startswith("Datetime") and "UTC" not in str(dtype):
                    try:
                        m = m.with_columns(pl.col(c).dt.convert_time_zone("UTC"))
                    except Exception:
                        m = m.with_columns(pl.col(c).dt.replace_time_zone("UTC"))
        m_for_feat = m.select(
            "condition_id", "resolved_at", "closed_at", "end_date"
        ).unique(subset=["condition_id"], keep="first")

    features = compute_trade_features(resolved, m_for_feat)
    # Attach token_won / event_id back onto features
    attach = [c for c in ("trade_id", "token_won", "event_id", "side", "whale_ratio") if c in resolved.columns]
    # whale_ratio comes from features; attach token_won/event_id/side
    attach = [c for c in ("trade_id", "token_won", "event_id", "side") if c in resolved.columns]
    if attach and not features.is_empty():
        features = features.join(
            resolved.select(attach).unique(subset=["trade_id"], keep="first"),
            on="trade_id",
            how="left",
        )

    features_path = curated / "trade_features_canonical.parquet"
    features.write_parquet(features_path, compression="snappy")

    event_col = "event_id" if "event_id" in resolved.columns and resolved["event_id"].null_count() < resolved.height else "condition_id"
    n_events = int(resolved[event_col].drop_nulls().n_unique()) if not resolved.is_empty() else 0
    n_conditions = int(resolved["condition_id"].n_unique()) if not resolved.is_empty() else 0

    return {
        "source": source,
        "n_rows_raw": raw.height,
        "n_rows_canonical": canonical.height,
        "n_rows_resolved": n_after_res,
        "n_rows_unresolved_dropped": n_before_res - n_after_res,
        "n_rows_features": features.height,
        "n_independent_events": n_events,
        "n_conditions": n_conditions,
        "event_col": event_col,
        "date_min": str(resolved["traded_at"].min()) if not resolved.is_empty() else None,
        "date_max": str(resolved["traded_at"].max()) if not resolved.is_empty() else None,
        "audit_before": audit_before,
        "audit_after": audit_after,
        "trades_path": str(trades_path),
        "features_path": str(features_path),
        "fee_regime_note": "zero_fee_historical for 2022-2023 (exact)",
    }
