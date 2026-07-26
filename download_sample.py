#!/usr/bin/env python3
"""Download a ~100K-row public Polymarket sample into data/raw/ for Phase 1–3.

Sources (HuggingFace SII-WANGZJ/Polymarket_data):
  - markets.parquet — download full file, write a resolved subset locally
  - trades.parquet (~28GB) — stream first ~100K rows only
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent
RAW_TRADES = ROOT / "data" / "raw" / "trades"
RAW_MARKETS = ROOT / "data" / "raw" / "markets"
CACHE = ROOT / "data" / "cache" / "hf"
TRADE_SAMPLE_N = 100_000
MARKET_SAMPLE_N = 80_000
REPO = "SII-WANGZJ/Polymarket_data"


def _ensure_dirs() -> None:
    RAW_TRADES.mkdir(parents=True, exist_ok=True)
    RAW_MARKETS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)


def download_markets() -> Path:
    from huggingface_hub import hf_hub_download

    print(f"Downloading markets.parquet from {REPO} ...", flush=True)
    path = hf_hub_download(
        repo_id=REPO,
        filename="markets.parquet",
        repo_type="dataset",
        local_dir=str(CACHE),
    )
    src = Path(path)
    print(f"  cached at {src} ({src.stat().st_size / 1e6:.1f} MB)", flush=True)
    return src


def download_trades_slice(n: int = TRADE_SAMPLE_N) -> Path:
    from datasets import load_dataset

    out = CACHE / f"trades_sample_{n}.parquet"
    if out.exists() and out.stat().st_size > 1_000_000:
        print(f"Reusing cached trade sample: {out}", flush=True)
        return out

    print(f"Streaming first {n:,} trades from {REPO}/trades.parquet ...", flush=True)
    ds = load_dataset(
        REPO,
        data_files="trades.parquet",
        split="train",
        streaming=True,
    )

    rows: list[dict] = []
    for i, row in enumerate(ds):
        rows.append(dict(row))
        if (i + 1) % 10_000 == 0:
            print(f"  ... {i + 1:,} rows", flush=True)
        if i + 1 >= n:
            break

    if not rows:
        raise RuntimeError("No trade rows streamed from HuggingFace dataset")

    df = pl.DataFrame(rows)
    df.write_parquet(out, compression="snappy")
    print(f"  wrote {out} ({df.height:,} rows, {out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return out


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
        seq = list(prices)
        floats = [float(x) for x in seq]
    except Exception:
        return None
    if not floats or max(floats) < 0.99:
        return None
    idx = floats.index(max(floats))
    if idx == 0:
        return "Yes"
    if idx == 1:
        return "No"
    return str(idx)


def normalize_markets_for_ingest(src: Path) -> Path:
    df = pl.read_parquet(src)
    print(f"markets columns: {df.columns}", flush=True)
    print(f"markets rows: {df.height:,}", flush=True)

    # HF schema: condition_id, token1/token2, answer1/answer2, outcome_prices, ...
    out = df.select(
        [
            pl.col("condition_id").cast(pl.Utf8).alias("conditionId"),
            pl.col("question").cast(pl.Utf8).alias("question"),
            pl.col("slug").cast(pl.Utf8).alias("slug") if "slug" in df.columns else pl.lit(None).alias("slug"),
            pl.col("token1").cast(pl.Utf8).alias("token_yes"),
            pl.col("token2").cast(pl.Utf8).alias("token_no"),
            pl.col("closed").alias("closed") if "closed" in df.columns else pl.lit(False).alias("closed"),
            pl.col("volume").cast(pl.Float64, strict=False).alias("volume")
            if "volume" in df.columns
            else pl.lit(None).alias("volume"),
            pl.col("neg_risk").alias("negRisk") if "neg_risk" in df.columns else pl.lit(False).alias("negRisk"),
            pl.col("end_date").alias("endDate") if "end_date" in df.columns else pl.lit(None).alias("endDate"),
            pl.col("outcome_prices").alias("outcomePrices")
            if "outcome_prices" in df.columns
            else pl.lit(None).alias("outcomePrices"),
        ]
    )

    # Derive winningOutcome from terminal outcome_prices when available
    winners = (
        out["outcomePrices"].map_elements(_winner_from_outcome_prices, return_dtype=pl.Utf8)
        if "outcomePrices" in out.columns
        else pl.Series("winningOutcome", [None] * out.height, dtype=pl.Utf8)
    )
    out = out.with_columns(winners.alias("winningOutcome"))
    out = out.with_columns(
        (pl.col("winningOutcome").is_not_null()).alias("resolved")
    )

    # Prefer resolved closed markets for backtests
    resolved = out.filter(pl.col("resolved") == True)  # noqa: E712
    print(f"  resolved markets: {resolved.height:,}", flush=True)
    if resolved.height >= 1_000:
        out = resolved
    else:
        closed = out.filter(pl.col("closed") == True)  # noqa: E712
        print(f"  falling back to closed markets: {closed.height:,}", flush=True)
        out = closed if closed.height else out

    # Cap size for local ingest speed; keep highest-volume first
    if "volume" in out.columns:
        out = out.sort("volume", descending=True, nulls_last=True)
    out = out.head(MARKET_SAMPLE_N)

    dest = RAW_MARKETS / "hf_markets_sample.parquet"
    out.write_parquet(dest, compression="snappy")
    print(f"  wrote {dest} ({out.height:,} markets)", flush=True)
    return dest


def normalize_trades_for_ingest(src: Path, market_condition_ids: set[str]) -> Path:
    df = pl.read_parquet(src)
    print(f"trades columns: {df.columns}", flush=True)
    print(f"trades rows: {df.height:,}", flush=True)

    # Probe a few likely schemas from this HF dump
    def pick(*names: str) -> str | None:
        for n in names:
            if n in df.columns:
                return n
        return None

    cond = pick("condition_id", "conditionId", "market", "market_id")
    asset = pick("token_id", "tokenId", "asset", "asset_id", "maker_asset_id", "taker_asset_id")
    price = pick("price", "fill_price")
    size = pick("token_amount", "size", "amount", "shares", "taker_amount", "maker_amount")
    ts = pick("timestamp", "ts", "block_timestamp", "trade_time", "created_at")
    side = pick("taker_direction", "maker_direction", "side", "taker_side")
    wallet = pick("maker", "taker", "trader", "proxyWallet")
    tx = pick("transaction_hash", "tx_hash", "transactionHash")
    outcome = pick("outcome", "answer", "nonusdc_side")

    if not all([cond, asset, price, size, ts]):
        raise RuntimeError(
            "Trade sample missing required fields. "
            f"cond={cond} asset={asset} price={price} size={size} ts={ts} cols={df.columns}"
        )

    out = df.select(
        [
            pl.col(cond).cast(pl.Utf8).alias("conditionId"),
            pl.col(asset).cast(pl.Utf8).alias("asset"),
            pl.col(price).cast(pl.Float64, strict=False).alias("price"),
            pl.col(size).cast(pl.Float64, strict=False).alias("size"),
            pl.col(ts).alias("timestamp"),
            (
                pl.col(side).cast(pl.Utf8).alias("side")
                if side
                else pl.lit("BUY").alias("side")
            ),
            (
                pl.col(wallet).cast(pl.Utf8).alias("proxyWallet")
                if wallet
                else pl.lit(None).alias("proxyWallet")
            ),
            (
                pl.col(tx).cast(pl.Utf8).alias("transactionHash")
                if tx
                else pl.lit(None).alias("transactionHash")
            ),
            (
                pl.col(outcome).cast(pl.Utf8).alias("outcome")
                if outcome
                else pl.lit(None).alias("outcome")
            ),
        ]
    )

    before = out.height
    overlapped = out.filter(pl.col("conditionId").is_in(list(market_condition_ids)))
    print(f"  trades overlapping sampled markets: {overlapped.height:,} / {before:,}", flush=True)
    if overlapped.height >= 10_000:
        out = overlapped
    else:
        print("  keeping unfiltered trade head (sparse market overlap)", flush=True)

    out = out.head(TRADE_SAMPLE_N)
    dest = RAW_TRADES / "hf_trades_sample.parquet"
    out.write_parquet(dest, compression="snappy")
    print(f"  wrote {dest} ({out.height:,} rows)", flush=True)
    return dest


def main() -> int:
    _ensure_dirs()
    markets_src = download_markets()
    markets_out = normalize_markets_for_ingest(markets_src)

    mdf = pl.read_parquet(markets_out)
    cids = set(mdf["conditionId"].drop_nulls().to_list())
    print(f"market condition ids for filter: {len(cids):,}", flush=True)

    trades_src = download_trades_slice(TRADE_SAMPLE_N)
    normalize_trades_for_ingest(trades_src, cids)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
