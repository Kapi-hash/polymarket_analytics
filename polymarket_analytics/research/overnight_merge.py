"""Merge year-sharded overnight backfills into an outcome-ready canonical lake."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from polymarket_analytics.research.canonical_lake import _enrich_markets_for_resolution
from polymarket_analytics.research.duplicates import canonicalize_trades
from polymarket_analytics.research.historical_fees import lookup_fee_regime


def _read_parquets(paths: list[Path], *, schema: dict[str, pl.DataType] | None = None) -> pl.DataFrame:
    existing = [path for path in paths if path.exists()]
    return (
        pl.concat([pl.read_parquet(path) for path in existing], how="diagonal_relaxed")
        if existing
        else pl.DataFrame(schema=schema)
    )


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
    categories = trades["category"].to_list() if "category" in trades.columns else [None] * trades.height
    regimes = [
        lookup_fee_regime(value, category=cat)
        for value, cat in zip(trades["traded_at"].to_list(), categories, strict=False)
    ]
    return trades.with_columns(
        pl.Series("fee_regime", [r["fee_regime"] for r in regimes]),
        pl.Series("fee_confidence", [r["fee_confidence"] for r in regimes]),
        pl.Series("fee_model_version", [r["fee_model_version"] for r in regimes]),
        pl.Series("fee_rate", [r["taker_rate"] for r in regimes], dtype=pl.Float64),
        pl.Series("fee_category", [r["category"] for r in regimes]),
    )


def _coalesce_join(base: pl.DataFrame, incoming: pl.DataFrame, key: str) -> pl.DataFrame:
    """Left-join incoming onto base, preferring non-null base values for overlapping columns."""
    if base.is_empty():
        return incoming
    if incoming.is_empty():
        return base
    overlap = [c for c in incoming.columns if c != key and c in base.columns]
    renamed = incoming.rename({c: f"{c}__new" for c in overlap})
    joined = base.join(renamed, on=key, how="left")
    for col in overlap:
        joined = joined.with_columns(pl.coalesce([pl.col(col), pl.col(f"{col}__new")]).alias(col)).drop(f"{col}__new")
    return joined


def _markets_for_features(markets: pl.DataFrame) -> pl.DataFrame:
    if markets.is_empty():
        return markets
    cols = ["condition_id"]
    for c in ("resolved_at", "closed_at", "closed_time", "end_date"):
        if c in markets.columns:
            cols.append(c)
    out = markets.select(cols)
    if "closed_at" not in out.columns and "closed_time" in out.columns:
        out = out.with_columns(pl.col("closed_time").alias("closed_at"))
    if "closed_at" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("closed_at"))
    if "resolved_at" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("resolved_at"))
    if "end_date" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("end_date"))
    return out.select("condition_id", "resolved_at", "closed_at", "end_date")


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


def _fingerprint(df: pl.DataFrame, cols: list[str]) -> str:
    use = [c for c in cols if c in df.columns]
    if df.is_empty() or not use:
        return hashlib.sha256(b"empty").hexdigest()
    sample = df.select(use).head(1000).write_csv()
    return hashlib.sha256(sample.encode()).hexdigest()


def merge_expanded_lake(
    data_root: Path,
    year_dirs: list[Path],
    existing_canonical: Path | None,
    out_dir: Path,
    *,
    baseline_features: Path | None = None,
    require_new_features: bool = True,
) -> dict[str, Any]:
    """Combine annual partitions, deduplicate fills, and attach resolutions/fees."""
    data_root, out_dir = Path(data_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_canonical = existing_canonical or data_root / "curated" / "trades_canonical.parquet"
    baseline_features = baseline_features or data_root / "curated" / "trade_features_canonical.parquet"

    annual_trades = _read_parquets(
        [Path(d) / "curated" / "trades.parquet" for d in year_dirs],
        schema={"trade_id": pl.Utf8, "condition_id": pl.Utf8, "token_id": pl.Utf8},
    )
    annual_markets = _read_parquets(
        [Path(d) / "curated" / "markets.parquet" for d in year_dirs],
        schema={"condition_id": pl.Utf8},
    )

    baseline = pl.read_parquet(existing_canonical) if existing_canonical.exists() else pl.DataFrame()
    baseline_n = baseline.height
    inputs = [x for x in (baseline, annual_trades) if not x.is_empty()]
    raw_trades = pl.concat(inputs, how="diagonal_relaxed") if inputs else pl.DataFrame()
    canonical = canonicalize_trades(raw_trades) if not raw_trades.is_empty() else raw_trades
    markets = _prepare_markets(annual_markets)

    # Prefer baseline event metadata; coalesce annual market joins without nulling.
    if not canonical.is_empty() and "event_id" in canonical.columns:
        # Keep existing event_id; fill only nulls from markets.
        pass
    if not canonical.is_empty() and not markets.is_empty() and "condition_id" in canonical.columns:
        keep = [
            c
            for c in (
                "condition_id",
                "event_id",
                "event_slug",
                "winning_token_id",
                "winning_outcome",
                "resolved",
                "resolved_at",
                "end_date",
                "closed_time",
                "question",
                "slug",
                "category",
            )
            if c in markets.columns
        ]
        market_join = markets.select(keep).unique(subset=["condition_id"], keep="last")
        # Coalesce per-column instead of dropping populated baseline fields.
        renamed = {
            c: f"{c}__mkt"
            for c in market_join.columns
            if c != "condition_id" and c in canonical.columns
        }
        mj = market_join.rename(renamed) if renamed else market_join
        canonical = canonical.join(mj, on="condition_id", how="left")
        for old, new in renamed.items():
            canonical = canonical.with_columns(pl.coalesce([pl.col(old), pl.col(new)]).alias(old)).drop(new)
        # Columns only on market side
        for c in market_join.columns:
            if c != "condition_id" and c not in renamed and f"{c}" not in canonical.columns:
                pass
        if "winning_token_id" in canonical.columns and "token_id" in canonical.columns:
            canonical = canonical.with_columns((pl.col("token_id") == pl.col("winning_token_id")).alias("token_won"))

    # Ensure event_id is never silently replaced by condition_id elsewhere.
    if not canonical.is_empty() and "event_id" not in canonical.columns:
        canonical = canonical.with_columns(pl.lit(None).cast(pl.Utf8).alias("event_id"))

    canonical = _attach_fees(canonical)

    from polymarket_analytics.features import compute_trade_features

    required = {"trade_id", "token_id", "traded_at", "price", "size"}
    missing = sorted(required - set(canonical.columns))
    if missing:
        raise RuntimeError(f"feature generation blocked: missing columns {missing}")
    if canonical.is_empty():
        raise RuntimeError("feature generation blocked: empty canonical trades")

    market_dates = _markets_for_features(markets) if not markets.is_empty() else None
    # Also try baseline markets lake if annual markets lack resolution fields.
    features = compute_trade_features(canonical, market_dates)
    if "token_won" in canonical.columns:
        features = features.join(
            canonical.select([c for c in ("trade_id", "token_won", "event_id", "condition_id", "side", "fee_regime", "fee_confidence", "fee_model_version", "category") if c in canonical.columns]),
            on="trade_id",
            how="left",
        )
    feature_rows = features.height
    if feature_rows == 0:
        raise RuntimeError("feature generation produced zero rows")

    baseline_feat_n = 0
    if baseline_features.exists():
        baseline_feat_n = pl.read_parquet(baseline_features).height
    new_feature_rows = feature_rows - baseline_feat_n
    if require_new_features and new_feature_rows <= 0:
        raise RuntimeError(
            f"no new feature rows relative to baseline ({feature_rows} vs baseline {baseline_feat_n})"
        )

    features.write_parquet(out_dir / "trade_features_canonical_expanded.parquet", compression="snappy")
    canonical.write_parquet(out_dir / "trades_canonical_expanded.parquet", compression="snappy")
    markets.write_parquet(out_dir / "markets_canonical_expanded.parquet", compression="snappy")

    fee_counts = (
        {str(k): int(v) for k, v in canonical.group_by("fee_regime").len().rows()}
        if not canonical.is_empty() and "fee_regime" in canonical.columns
        else {}
    )
    (out_dir / "fee_regime_summary.json").write_text(json.dumps(fee_counts, indent=2), encoding="utf-8")

    event_populated = 0.0
    n_events = 0
    if "event_id" in canonical.columns and not canonical.is_empty():
        event_populated = 1.0 - (canonical["event_id"].null_count() / canonical.height)
        n_events = int(canonical["event_id"].drop_nulls().n_unique())

    coverage = {
        "status": "ok",
        "baseline_trade_rows": baseline_n,
        "n_trades": canonical.height,
        "rows_added": canonical.height - baseline_n,
        "n_events": n_events,
        "event_id_populated_frac": event_populated,
        "n_conditions": int(canonical["condition_id"].drop_nulls().n_unique())
        if not canonical.is_empty() and "condition_id" in canonical.columns
        else 0,
        "date_min": str(canonical["traded_at"].min()) if not canonical.is_empty() and "traded_at" in canonical.columns else None,
        "date_max": str(canonical["traded_at"].max()) if not canonical.is_empty() and "traded_at" in canonical.columns else None,
        "fee_regime_counts": fee_counts,
        "features": {
            "rows": feature_rows,
            "baseline_rows": baseline_feat_n,
            "rows_added": new_feature_rows,
            "status": "computed",
            "path": str(out_dir / "trade_features_canonical_expanded.parquet"),
        },
        "fingerprints": {
            "trades": _fingerprint(canonical, ["trade_id", "condition_id", "traded_at"]),
            "features": _fingerprint(features, ["trade_id", "token_id", "traded_at"]),
        },
        "n_exclusions": _write_exclusions(year_dirs, out_dir / "exclusions.csv"),
        "used_baseline_fallback": False,
    }
    (out_dir / "coverage_report.json").write_text(json.dumps(coverage, indent=2, default=str), encoding="utf-8")
    return coverage
