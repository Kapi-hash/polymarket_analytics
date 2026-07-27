"""Prospective L2 book collector for Polymarket CLOB."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import polars as pl

WS_MARKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
REST_BOOK_URL = "https://clob.polymarket.com/book?token_id={token_id}"


@dataclass
class SessionHealth:
    connected: bool = False
    messages_received: int = 0
    gaps_detected: int = 0
    reconnects: int = 0
    last_message_at: float | None = None
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_book_message(raw: dict[str, Any]) -> dict[str, Any] | None:
    """
    Normalize a websocket book/trade message to best bid/ask + metadata.

    Handles price_change and book snapshot shapes.
    """
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
        "event_type": event_type,
        "asset_id": asset_id,
        "best_bid": float(best_bid) if best_bid is not None else None,
        "best_ask": float(best_ask) if best_ask is not None else None,
        "timestamp": ts,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "raw_keys": list(raw.keys()),
    }


def parse_trade_message(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Extract trade print fields from websocket payload."""
    if not raw:
        return None
    event_type = raw.get("event_type") or raw.get("type")
    if event_type not in {"last_trade_price", "trade", "match"}:
        price = raw.get("price")
        if price is None:
            return None
    price = raw.get("price") or raw.get("last_trade_price")
    if price is None:
        return None
    return {
        "asset_id": raw.get("asset_id") or raw.get("token_id"),
        "price": float(price),
        "size": float(raw.get("size") or raw.get("amount") or 0.0),
        "side": raw.get("side"),
        "timestamp": raw.get("timestamp") or raw.get("ts"),
        "event_type": event_type,
        "received_at": datetime.now(timezone.utc).isoformat(),
    }


def detect_sequence_gap(prev_seq: int | None, seq: int | None) -> bool:
    """Return True if sequence jumped by more than 1."""
    if prev_seq is None or seq is None:
        return False
    return seq - prev_seq > 1


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
        "token_id": token_id,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "n_bid_levels": len(bids),
        "n_ask_levels": len(asks),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


async def _collect_ws(
    token_ids: list[str],
    *,
    duration_sec: float,
    out_raw: Path,
    out_normalized: Path,
    health: SessionHealth,
) -> None:
    import websockets

    out_raw.mkdir(parents=True, exist_ok=True)
    out_normalized.mkdir(parents=True, exist_ok=True)
    raw_path = out_raw / f"session_{int(time.time())}.jsonl"
    norm_rows: list[dict[str, Any]] = []
    prev_seq: int | None = None
    flush_index = 0
    flush_every = 500

    def flush_normalized() -> None:
        """Persist bounded chunks so a long collection does not retain all rows."""
        nonlocal flush_index
        if not norm_rows:
            return
        path = out_normalized / f"session_{int(time.time())}_{flush_index:05d}.parquet"
        pl.DataFrame(norm_rows).write_parquet(path, compression="snappy")
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
                        continue

                    if isinstance(payload, list):
                        items = payload
                    else:
                        items = [payload]

                    with raw_path.open("a") as fh:
                        for item in items:
                            fh.write(json.dumps(item) + "\n")
                            seq = item.get("sequence") or item.get("seq")
                            if isinstance(seq, int) and detect_sequence_gap(prev_seq, seq):
                                health.gaps_detected += 1
                            if isinstance(seq, int):
                                prev_seq = seq
                            book = parse_book_message(item)
                            trade = parse_trade_message(item)
                            if book:
                                norm_rows.append(book)
                            if trade:
                                norm_rows.append({**trade, "record_type": "trade"})
                            if len(norm_rows) >= flush_every:
                                flush_normalized()
        except Exception:
            health.reconnects += 1
            health.connected = False
            await asyncio.sleep(1.0)

    flush_normalized()


def run_smoke_test(
    token_ids: list[str],
    *,
    duration_sec: float = 5.0,
    max_tokens: int = 3,
    data_root: Path | str = "data",
) -> dict[str, Any]:
    """
    Smoke test: connect, collect raw + normalized for ``duration_sec``.

    Limits to ``max_tokens`` subscriptions.
    """
    data_root = Path(data_root)
    tokens = token_ids[:max_tokens]
    health = SessionHealth()
    out_raw = data_root / "books" / "raw"
    out_norm = data_root / "books" / "normalized"

    asyncio.run(
        _collect_ws(
            tokens,
            duration_sec=duration_sec,
            out_raw=out_raw,
            out_normalized=out_norm,
            health=health,
        )
    )
    return {
        "token_ids": tokens,
        "duration_sec": duration_sec,
        "health": health.to_dict(),
        "raw_dir": str(out_raw),
        "normalized_dir": str(out_norm),
    }
