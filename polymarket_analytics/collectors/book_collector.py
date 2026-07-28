"""Prospective L2 book collector for Polymarket CLOB."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import polars as pl

WS_MARKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
REST_BOOK_URL = "https://clob.polymarket.com/book?token_id={token_id}"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"

NORMALIZED_SCHEMA: dict[str, pl.DataType] = {
    "event_type": pl.Utf8,
    "asset_id": pl.Utf8,
    "best_bid": pl.Float64,
    "best_ask": pl.Float64,
    "price": pl.Float64,
    "size": pl.Float64,
    "side": pl.Utf8,
    "timestamp": pl.Utf8,
    "received_at": pl.Utf8,
    "record_type": pl.Utf8,
    "raw_keys": pl.List(pl.Utf8),
    "session_id": pl.Utf8,
    "parse_ok": pl.Boolean,
}


@dataclass
class SessionHealth:
    connected: bool = False
    messages_received: int = 0
    gaps_detected: int = 0
    reconnects: int = 0
    malformed_messages: int = 0
    normalized_rows: int = 0
    raw_rows: int = 0
    last_message_at: float | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    terminal_error: str | None = None
    emitting_tokens: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_book_message(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a websocket book/trade message to best bid/ask + metadata."""
    if not raw:
        return None

    event_type = raw.get("event_type") or raw.get("type") or raw.get("channel")
    asset_id = raw.get("asset_id") or raw.get("token_id")
    ts = raw.get("timestamp") or raw.get("ts")
    best_bid = raw.get("best_bid")
    best_ask = raw.get("best_ask")

    bids = raw.get("bids") or raw.get("buys")
    asks = raw.get("asks") or raw.get("sells")
    if best_bid is None and isinstance(bids, list) and bids:
        best_bid = max(float(b.get("price", b[0] if isinstance(b, (list, tuple)) else 0)) for b in bids)
    if best_ask is None and isinstance(asks, list) and asks:
        best_ask = min(float(a.get("price", a[0] if isinstance(a, (list, tuple)) else 1)) for a in asks)

    if best_bid is None and best_ask is None and event_type not in {"price_change", "book", "last_trade_price"}:
        return None

    return {
        "event_type": str(event_type) if event_type is not None else None,
        "asset_id": str(asset_id) if asset_id is not None else None,
        "best_bid": float(best_bid) if best_bid is not None else None,
        "best_ask": float(best_ask) if best_ask is not None else None,
        "price": None,
        "size": None,
        "side": None,
        "timestamp": str(ts) if ts is not None else None,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "record_type": "book",
        "raw_keys": [str(k) for k in raw.keys()],
        "parse_ok": True,
    }


def parse_trade_message(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Extract trade print fields from websocket payload."""
    if not raw:
        return None
    event_type = raw.get("event_type") or raw.get("type")
    if event_type not in {"last_trade_price", "trade", "match"}:
        if raw.get("price") is None:
            return None
    price = raw.get("price") or raw.get("last_trade_price")
    if price is None:
        return None
    return {
        "event_type": str(event_type) if event_type is not None else None,
        "asset_id": str(raw.get("asset_id") or raw.get("token_id") or ""),
        "best_bid": None,
        "best_ask": None,
        "price": float(price),
        "size": float(raw.get("size") or raw.get("amount") or 0.0),
        "side": str(raw.get("side")) if raw.get("side") is not None else None,
        "timestamp": str(raw.get("timestamp") or raw.get("ts") or ""),
        "received_at": datetime.now(timezone.utc).isoformat(),
        "record_type": "trade",
        "raw_keys": [str(k) for k in raw.keys()],
        "parse_ok": True,
    }


def detect_sequence_gap(prev_seq: int | None, seq: int | None) -> bool:
    """Return True if sequence jumped by more than 1."""
    if prev_seq is None or seq is None:
        return False
    return seq - prev_seq > 1


def write_normalized_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Write normalized rows with an explicit schema so large asset IDs stay strings."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    aligned = []
    for row in rows:
        aligned.append({key: row.get(key) for key in NORMALIZED_SCHEMA})
    pl.DataFrame(aligned, schema=NORMALIZED_SCHEMA).write_parquet(path, compression="snappy")


def rebuild_normalized_from_raw(raw_path: Path, out_path: Path, *, session_id: str) -> dict[str, Any]:
    """Rebuild normalized parquet from raw JSONL (recovery path)."""
    rows: list[dict[str, Any]] = []
    malformed = 0
    with raw_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(item, list):
                items = item
            else:
                items = [item]
            for payload in items:
                if not isinstance(payload, dict):
                    malformed += 1
                    continue
                book = parse_book_message(payload)
                trade = parse_trade_message(payload)
                if book:
                    rows.append({**book, "session_id": session_id})
                if trade:
                    rows.append({**trade, "session_id": session_id})
    write_normalized_rows(rows, out_path)
    return {"raw_path": str(raw_path), "out_path": str(out_path), "normalized_rows": len(rows), "malformed": malformed}


def rest_book_snapshot(token_id: str, *, timeout_s: float = 10.0) -> dict[str, Any]:
    """Fetch REST L2 snapshot for one token."""
    url = REST_BOOK_URL.format(token_id=token_id)
    with urlopen(url, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode())
    bids = data.get("bids") or []
    asks = data.get("asks") or []
    best_bid = max((float(b["price"]) for b in bids), default=None)
    best_ask = min((float(a["price"]) for a in asks), default=None)
    return {
        "token_id": str(token_id),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "n_bid_levels": len(bids),
        "n_ask_levels": len(asks),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def discover_active_token_ids(n_tokens: int = 100) -> list[str]:
    """Discover active CLOB token IDs from Gamma (requires User-Agent)."""
    req = Request(
        f"{GAMMA_MARKETS_URL}?active=true&closed=false&limit={min(max(n_tokens, 20), 100)}",
        headers={"User-Agent": "polymarket-analytics-research/0.8", "Accept": "application/json"},
    )
    with urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode())
    token_ids: list[str] = []
    markets = payload if isinstance(payload, list) else payload.get("markets") or payload.get("data") or []
    for market in markets:
        raw = market.get("clobTokenIds") or market.get("clob_token_ids")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = []
        if isinstance(raw, list):
            for tok in raw:
                token_ids.append(str(tok))
                if len(token_ids) >= n_tokens:
                    return token_ids
    return token_ids


async def _collect_ws(
    token_ids: list[str],
    *,
    duration_sec: float,
    out_raw: Path,
    out_normalized: Path,
    health: SessionHealth,
    session_id: str,
) -> None:
    import websockets

    out_raw.mkdir(parents=True, exist_ok=True)
    out_normalized.mkdir(parents=True, exist_ok=True)
    raw_path = out_raw / f"session_{session_id}.jsonl"
    norm_rows: list[dict[str, Any]] = []
    prev_seq: int | None = None
    flush_index = 0
    flush_every = 500
    emitting: set[str] = set()

    def flush_normalized() -> None:
        nonlocal flush_index
        if not norm_rows:
            return
        path = out_normalized / f"session_{session_id}_{flush_index:05d}.parquet"
        write_normalized_rows(norm_rows, path)
        health.normalized_rows += len(norm_rows)
        norm_rows.clear()
        flush_index += 1

    subscribe = {"assets_ids": token_ids, "type": "market"}
    deadline = time.time() + duration_sec

    while time.time() < deadline:
        try:
            async with websockets.connect(WS_MARKET_URL, ping_interval=20) as ws:
                health.connected = True
                await ws.send(json.dumps(subscribe))
                while time.time() < deadline:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        continue
                    health.messages_received += 1
                    health.last_message_at = time.time()
                    try:
                        payload = json.loads(msg)
                    except json.JSONDecodeError:
                        health.malformed_messages += 1
                        continue

                    items = payload if isinstance(payload, list) else [payload]
                    with raw_path.open("a") as fh:
                        for item in items:
                            if not isinstance(item, dict):
                                health.malformed_messages += 1
                                continue
                            fh.write(json.dumps(item) + "\n")
                            health.raw_rows += 1
                            seq = item.get("sequence") or item.get("seq")
                            if isinstance(seq, int) and detect_sequence_gap(prev_seq, seq):
                                health.gaps_detected += 1
                            if isinstance(seq, int):
                                prev_seq = seq
                            book = parse_book_message(item)
                            trade = parse_trade_message(item)
                            if book:
                                if book.get("asset_id"):
                                    emitting.add(str(book["asset_id"]))
                                norm_rows.append({**book, "session_id": session_id})
                            if trade:
                                if trade.get("asset_id"):
                                    emitting.add(str(trade["asset_id"]))
                                norm_rows.append({**trade, "session_id": session_id})
                            if len(norm_rows) >= flush_every:
                                flush_normalized()
        except Exception as exc:  # reconnect; preserve raw already flushed
            health.reconnects += 1
            health.connected = False
            health.terminal_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(1.0)

    flush_normalized()
    health.ended_at = time.time()
    health.emitting_tokens = sorted(emitting)


def run_collection(
    token_ids: list[str],
    *,
    duration_sec: float = 5.0,
    max_tokens: int = 100,
    data_root: Path | str = "data",
    session_id: str | None = None,
    run_dir_name: str | None = None,
) -> dict[str, Any]:
    """Collect raw + normalized L2 for ``duration_sec`` into a clean run directory."""
    data_root = Path(data_root)
    session_id = session_id or uuid.uuid4().hex[:12]
    run_dir = data_root / "books" / "runs" / (run_dir_name or session_id)
    tokens = [str(t) for t in token_ids[:max_tokens]]
    health = SessionHealth()
    out_raw = run_dir / "raw"
    out_norm = run_dir / "normalized"

    try:
        asyncio.run(
            _collect_ws(
                tokens,
                duration_sec=duration_sec,
                out_raw=out_raw,
                out_normalized=out_norm,
                health=health,
                session_id=session_id,
            )
        )
    except Exception as exc:  # noqa: BLE001
        health.terminal_error = f"{type(exc).__name__}: {exc}"
        health.ended_at = time.time()

    actual_duration = (health.ended_at or time.time()) - health.started_at
    emitting = health.emitting_tokens
    report = {
        "ok": True,
        "session_id": session_id,
        "run_dir": str(run_dir),
        "requested_tokens": max_tokens,
        "selected_tokens": tokens,
        "emitting_tokens": emitting,
        "duration_sec_requested": duration_sec,
        "duration_sec_actual": actual_duration,
        "health": health.to_dict(),
        "raw_dir": str(out_raw),
        "normalized_dir": str(out_norm),
    }

    # Fail closed on material shortfalls / crashes.
    min_ok_duration = max(1.0, duration_sec * 0.85)
    if health.terminal_error and health.raw_rows == 0:
        report["ok"] = False
        report["error"] = health.terminal_error
    elif health.raw_rows == 0:
        report["ok"] = False
        report["error"] = "no raw rows collected"
    elif health.normalized_rows == 0:
        report["ok"] = False
        report["error"] = "no normalized rows collected"
    elif actual_duration < min_ok_duration and duration_sec >= 30:
        report["ok"] = False
        report["error"] = (
            f"collection ended early: actual={actual_duration:.1f}s requested={duration_sec:.1f}s"
        )

    health_path = run_dir / "collector_health.json"
    health_path.parent.mkdir(parents=True, exist_ok=True)
    health_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def run_smoke_test(
    token_ids: list[str],
    *,
    duration_sec: float = 5.0,
    max_tokens: int = 3,
    data_root: Path | str = "data",
) -> dict[str, Any]:
    """Backward-compatible smoke entrypoint that writes into a unique run directory."""
    return run_collection(
        token_ids,
        duration_sec=duration_sec,
        max_tokens=max_tokens,
        data_root=data_root,
    )
