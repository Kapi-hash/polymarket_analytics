"""Resumable, year-sharded historical Gamma/Data API backfill."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl

UA = "polymarket-analytics-research/0.7"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
DATA_TRADES_URL = "https://data-api.polymarket.com/trades"


@dataclass(frozen=True)
class RetrievedPage:
    payload: list[dict[str, Any]]
    url: str
    retrieved_at: str
    response_sha256: str


def http_get_json(url: str, *, max_retries: int = 6) -> Any:
    """Fetch JSON with bounded exponential retry and courteous pacing."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            request = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            time.sleep(random.uniform(0.2, 0.5))
            return payload
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(min(20.0, 0.5 * (2**attempt)) + random.uniform(0.0, 0.5))
    raise RuntimeError(f"GET failed after {max_retries} attempts: {url}: {last_error}") from last_error


def _as_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "results", "trades", "markets"):
            if isinstance(payload.get(key), list):
                return [row for row in payload[key] if isinstance(row, dict)]
    return []


def _parse_datetime(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        # APIs generally return seconds; tolerate milliseconds.
        seconds = float(value) / 1000 if float(value) > 10_000_000_000 else float(value)
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    text = str(value).strip()
    try:
        if text.isdigit():
            return _parse_datetime(int(text))
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(text[:10], fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
    return None


def _market_matches_year(market: dict[str, Any], year: int) -> bool:
    """Match a market when either Gamma closure/end date is in ``year``."""
    return any(
        (parsed := _parse_datetime(market.get(field))) is not None and parsed.year == year
        for field in ("closedTime", "closed_time", "endDate", "end_date")
    )


def _condition_id(market: dict[str, Any]) -> str | None:
    value = market.get("conditionId") or market.get("condition_id")
    return str(value) if value not in (None, "") else None


def _trade_id(trade: dict[str, Any], index: int) -> str:
    tx_hash = trade.get("transactionHash") or trade.get("transaction_hash") or trade.get("tx_hash") or ""
    log_index = trade.get("logIndex") or trade.get("log_index") or trade.get("index") or index
    return f"{tx_hash}_{log_index}"


def _json_value(value: object) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _read_existing(path: Path) -> pl.DataFrame | None:
    return pl.read_parquet(path) if path.exists() else None


def _market_schema() -> dict[str, pl.DataType]:
    return {
        "condition_id": pl.Utf8, "question": pl.Utf8, "slug": pl.Utf8, "event_id": pl.Utf8,
        "event_slug": pl.Utf8, "closed": pl.Boolean, "end_date": pl.Datetime("us", "UTC"),
        "closed_time": pl.Datetime("us", "UTC"), "volume": pl.Float64, "outcome_prices": pl.Utf8,
        "clob_token_ids": pl.Utf8, "category": pl.Utf8, "year": pl.Int64,
    }


def _trade_schema() -> dict[str, pl.DataType]:
    return {
        "trade_id": pl.Utf8, "condition_id": pl.Utf8, "token_id": pl.Utf8, "side": pl.Utf8,
        "price": pl.Float64, "size": pl.Float64, "notional": pl.Float64, "tx_hash": pl.Utf8,
        "traded_at": pl.Datetime("us", "UTC"), "wallet": pl.Utf8, "source_file": pl.Utf8,
        "year": pl.Int64,
    }


def _normalize_market(market: dict[str, Any], year: int) -> dict[str, Any]:
    return {
        "condition_id": _condition_id(market),
        "question": market.get("question"),
        "slug": market.get("slug"),
        "event_id": market.get("eventId") or market.get("event_id"),
        "event_slug": market.get("eventSlug") or market.get("event_slug"),
        "closed": market.get("closed"),
        "end_date": _parse_datetime(market.get("endDate") or market.get("end_date")),
        "closed_time": _parse_datetime(market.get("closedTime") or market.get("closed_time")),
        "volume": market.get("volume") or market.get("volumeNum"),
        "outcome_prices": _json_value(market.get("outcomePrices") or market.get("outcome_prices")),
        "clob_token_ids": _json_value(market.get("clobTokenIds") or market.get("clob_token_ids")),
        "category": market.get("category"),
        "year": year,
    }


def _normalize_trade(
    trade: dict[str, Any], condition_id: str, source_file: str, year: int, index: int
) -> dict[str, Any]:
    price = trade.get("price")
    size = trade.get("size") or trade.get("tokenAmount") or trade.get("token_amount")
    try:
        notional = float(price) * float(size)
    except (TypeError, ValueError):
        notional = trade.get("amount") or trade.get("usdcSize") or trade.get("usdc_size")
    return {
        "trade_id": _trade_id(trade, index),
        "condition_id": condition_id,
        "token_id": str(trade.get("asset") or trade.get("asset_id") or trade.get("tokenId") or "") or None,
        "side": trade.get("side") or trade.get("takerDirection") or trade.get("taker_direction"),
        "price": price,
        "size": size,
        "notional": notional,
        "tx_hash": trade.get("transactionHash") or trade.get("transaction_hash") or trade.get("tx_hash"),
        "traded_at": _parse_datetime(trade.get("timestamp") or trade.get("tradedAt") or trade.get("createdAt")),
        "wallet": trade.get("proxyWallet") or trade.get("maker") or trade.get("taker") or trade.get("wallet"),
        "source_file": source_file,
        "year": year,
    }


def _write_exclusions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("condition_id", "reason", "detail")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def backfill_year(
    year: int,
    out_dir: Path,
    *,
    market_limit: int | None = None,
    trade_limit_per_market: int = 500,
    max_markets: int | None = None,
) -> dict[str, Any]:
    """Backfill closed markets resolved in one year, resuming completed markets."""
    out_dir = Path(out_dir)
    raw_markets = out_dir / "raw" / "markets"
    raw_trades = out_dir / "raw" / "trades"
    curated = out_dir / "curated"
    checkpoint_path = out_dir / "checkpoints" / "state.json"
    exclusions_path = out_dir / "reports" / "exclusions.csv"
    state = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {
        "market_offset": 0,
        "completed_condition_ids": [],
    }
    completed = set(state.get("completed_condition_ids", []))
    market_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    n_pages = 0
    n_seen = 0
    offset = int(state.get("market_offset", 0))
    selected = 0

    while True:
        if market_limit is not None and n_pages >= market_limit:
            break
        url = f"{GAMMA_MARKETS_URL}?{urlencode({'closed': 'true', 'limit': 100, 'offset': offset})}"
        try:
            payload = http_get_json(url)
        except Exception as exc:  # page-level failure is checkpointed for safe retry
            failures.append({"condition_id": "", "reason": "market_page_fetch_failed", "detail": str(exc)})
            break
        records = _as_records(payload)
        retrieved_at = datetime.now(timezone.utc).isoformat()
        body = json.dumps(payload, sort_keys=True).encode()
        _write_json(
            raw_markets / f"page_{n_pages:04d}.json",
            {"provenance": {"source": "gamma", "request_url": url, "retrieved_at": retrieved_at,
                            "response_sha256": hashlib.sha256(body).hexdigest(), "page_index": n_pages},
             "data": payload},
        )
        n_pages += 1
        offset += 100
        candidates = [m for m in records if _market_matches_year(m, year)]
        for market in candidates:
            if max_markets is not None and selected >= max_markets:
                break
            cid = _condition_id(market)
            if not cid:
                failures.append({"condition_id": "", "reason": "missing_condition_id", "detail": str(market.get("id", ""))})
                continue
            n_seen += 1
            market_rows.append(_normalize_market(market, year))
            if cid in completed:
                continue
            selected += 1
            try:
                for page_index, trade_offset in enumerate(range(0, trade_limit_per_market, 100)):
                    trade_url = f"{DATA_TRADES_URL}?{urlencode({'conditionId': cid, 'limit': 100, 'offset': trade_offset})}"
                    trade_payload = http_get_json(trade_url)
                    records_t = _as_records(trade_payload)
                    trade_path = raw_trades / f"{cid}_p{page_index:04d}.json"
                    body = json.dumps(trade_payload, sort_keys=True).encode()
                    _write_json(
                        trade_path,
                        {"provenance": {"source": "data-api", "request_url": trade_url,
                                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                                        "response_sha256": hashlib.sha256(body).hexdigest(),
                                        "page_index": page_index}, "data": trade_payload},
                    )
                    trade_rows.extend(
                        _normalize_trade(row, cid, str(trade_path.relative_to(out_dir)), year, trade_offset + i)
                        for i, row in enumerate(records_t)
                    )
                    if len(records_t) < 100:
                        break
                completed.add(cid)
            except Exception as exc:  # do not mark failed markets complete
                failures.append({"condition_id": cid, "reason": "trade_fetch_failed", "detail": str(exc)})
            state = {"market_offset": offset, "completed_condition_ids": sorted(completed)}
            _write_json(checkpoint_path, state)
        state = {"market_offset": offset, "completed_condition_ids": sorted(completed)}
        _write_json(checkpoint_path, state)
        if (max_markets is not None and selected >= max_markets) or not records or len(records) < 100:
            break

    existing_markets = _read_existing(curated / "markets.parquet")
    existing_trades = _read_existing(curated / "trades.parquet")
    markets = pl.DataFrame(market_rows) if market_rows else pl.DataFrame(schema=_market_schema())
    trades = pl.DataFrame(trade_rows) if trade_rows else pl.DataFrame(schema=_trade_schema())
    if existing_markets is not None:
        markets = pl.concat([existing_markets, markets], how="diagonal_relaxed")
    if existing_trades is not None:
        trades = pl.concat([existing_trades, trades], how="diagonal_relaxed")
    if not markets.is_empty() and "condition_id" in markets.columns:
        markets = markets.unique(subset=["condition_id"], keep="last")
    if not trades.is_empty() and "trade_id" in trades.columns:
        trades = trades.unique(subset=["trade_id"], keep="last")
    curated.mkdir(parents=True, exist_ok=True)
    markets.write_parquet(curated / "markets.parquet", compression="snappy")
    trades.write_parquet(curated / "trades.parquet", compression="snappy")
    _write_exclusions(exclusions_path, failures)
    summary = {
        "year": year, "n_market_pages": n_pages, "n_markets": markets.height,
        "n_trades": trades.height, "n_failures": len(failures), "n_candidates_seen": n_seen,
        "checkpoint_path": str(checkpoint_path),
    }
    _write_json(out_dir / "reports" / "year_quality.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill one year of Polymarket trades")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--market-limit", type=int)
    parser.add_argument("--max-markets", type=int)
    parser.add_argument("--trade-limit-per-market", type=int, default=500)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args(argv)
    if args.tiny:
        args.max_markets, args.trade_limit_per_market = 5, 50
    print(json.dumps(backfill_year(args.year, args.out_dir, market_limit=args.market_limit,
                                   max_markets=args.max_markets, trade_limit_per_market=args.trade_limit_per_market), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
