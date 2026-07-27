"""Merge year-sharded overnight backfills into an outcome-ready canonical lake."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import polars as pl

from polymarket_analytics.research.canonical_lake import _enrich_markets_for_resolution
from polymarket_analytics.research.duplicates import canonicalize_trades
from polymarket_analytics.research.historical_fees import lookup_fee_regime


def _read_parquets(paths: list[Path], *, schema: dict[str, pl.DataType] | None = None) -> pl.DataFrame:
    existing = [path for path in paths if path.exists()]
    return pl.concat([pl.read_parquet(path) for path in existing], how="diagonal_relaxed") if existing else pl.DataFrame(schema=schema)


def _parse_tokens(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return [str(item) for item in value] if isinstance(value, list) else []


def _prepare_markets(markets: pl.DataFrame) -> pl.DataFrame:
    if markets.is_empty():
        return markets
    out = markets.unique(subset=["condition_id"], keep="last") if "condition_id" in markets.columns else markets
    if "clob_token_ids" in out.columns:
        tokens = out["clob_token_ids"].map_elements(_parse_tokens, return_dtype=pl.List(pl.Utf8))
        out = out.with_columns(
            tokens.list.get(0, null_on_oob=True).alias("token_yes"),
            tokens.list.get(1, null_on_oob=True).alias("token_no"),
        )
    return _enrich_markets_for_resolution(out)


def _attach_fees(trades: pl.DataFrame) -> pl.DataFrame:
    if trades.is_empty() or "traded_at" not in trades.columns:
        return trades
    regimes = [
        lookup_fee_regime(value)
        for value in trades["traded_at"].to_list()
    ]
    return trades.with_columns(
        pl.Series("fee_regime", [r["fee_regime"] for r in regimes]),
        pl.Series("fee_confidence", [r["fee_confidence"] for r in regimes]),
        pl.Series("fee_model_version", [r["fee_model_version"] for r in regimes]),
        pl.Series("fee_rate", [r["taker_rate"] for r in regimes], dtype=pl.Float64),
    )


def _write_exclusions(year_dirs: list[Path], out_path: Path) -> int:
    rows: list[dict[str, str]] = []
    for year_dir in year_dirs:
        path = year_dir / "reports" / "exclusions.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("condition_id", "reason", "detail"))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def merge_expanded_lake(
    data_root: Path,
    year_dirs: list[Path],
    existing_canonical: Path | None,
    out_dir: Path,
) -> dict[str, Any]:
    """Combine annual partitions, deduplicate fills, and attach resolutions/fees."""
    data_root, out_dir = Path(data_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_canonical = existing_canonical or data_root / "curated" / "trades_canonical.parquet"
    annual_trades = _read_parquets(
        [Path(d) / "curated" / "trades.parquet" for d in year_dirs],
        schema={"trade_id": pl.Utf8, "condition_id": pl.Utf8, "token_id": pl.Utf8},
    )
    annual_markets = _read_parquets(
        [Path(d) / "curated" / "markets.parquet" for d in year_dirs],
        schema={"condition_id": pl.Utf8},
    )
    inputs = [annual_trades]
    if existing_canonical.exists():
        inputs.append(pl.read_parquet(existing_canonical))
    raw_trades = pl.concat(inputs, how="diagonal_relaxed") if any(not x.is_empty() for x in inputs) else pl.DataFrame()
    canonical = canonicalize_trades(raw_trades) if not raw_trades.is_empty() else raw_trades
    markets = _prepare_markets(annual_markets)

    if not canonical.is_empty() and not markets.is_empty() and "condition_id" in canonical.columns:
        keep = [c for c in ("condition_id", "event_id", "winning_token_id", "winning_outcome", "resolved",
                            "resolved_at", "end_date", "closed_time", "question", "slug", "category")
                if c in markets.columns]
        market_join = markets.select(keep).unique(subset=["condition_id"], keep="last")
        canonical = canonical.drop([c for c in market_join.columns if c != "condition_id" and c in canonical.columns])
        canonical = canonical.join(market_join, on="condition_id", how="left")
        if "winning_token_id" in canonical.columns and "token_id" in canonical.columns:
            canonical = canonical.with_columns((pl.col("token_id") == pl.col("winning_token_id")).alias("token_won"))
    canonical = _attach_fees(canonical)

    features_note = "not attempted"
    feature_rows = 0
    try:
        from polymarket_analytics.features import compute_trade_features

        required = {"trade_id", "token_id", "traded_at", "price", "size"}
        if required.issubset(canonical.columns):
            market_dates = markets.select([c for c in ("condition_id", "resolved_at", "end_date") if c in markets.columns]) if not markets.is_empty() else None
            features = compute_trade_features(canonical, market_dates)
            features.write_parquet(out_dir / "trade_features_canonical_expanded.parquet", compression="snappy")
            feature_rows, features_note = features.height, "computed"
        else:
            features_note = f"skipped: missing {sorted(required - set(canonical.columns))}"
    except Exception as exc:  # feature support must not block durable merged lake
        features_note = f"skipped: {type(exc).__name__}: {exc}"

    canonical.write_parquet(out_dir / "trades_canonical_expanded.parquet", compression="snappy")
    markets.write_parquet(out_dir / "markets_canonical_expanded.parquet", compression="snappy")
    fee_counts = (
        {str(k): int(v) for k, v in canonical.group_by("fee_regime").len().rows()}
        if not canonical.is_empty() and "fee_regime" in canonical.columns else {}
    )
    (out_dir / "fee_regime_summary.json").write_text(json.dumps(fee_counts, indent=2), encoding="utf-8")
    event_col = "event_id" if "event_id" in canonical.columns and canonical["event_id"].null_count() < canonical.height else "condition_id"
    coverage = {
        "n_trades": canonical.height,
        "n_events": int(canonical[event_col].drop_nulls().n_unique()) if not canonical.is_empty() and event_col in canonical.columns else 0,
        "n_conditions": int(canonical["condition_id"].drop_nulls().n_unique()) if not canonical.is_empty() and "condition_id" in canonical.columns else 0,
        "date_min": str(canonical["traded_at"].min()) if not canonical.is_empty() and "traded_at" in canonical.columns else None,
        "date_max": str(canonical["traded_at"].max()) if not canonical.is_empty() and "traded_at" in canonical.columns else None,
        "fee_regime_counts": fee_counts,
        "features": {"rows": feature_rows, "status": features_note},
        "n_exclusions": _write_exclusions(year_dirs, out_dir / "exclusions.csv"),
    }
    (out_dir / "coverage_report.json").write_text(json.dumps(coverage, indent=2, default=str), encoding="utf-8")
    return coverage
