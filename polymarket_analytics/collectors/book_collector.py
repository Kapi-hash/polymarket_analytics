"""Prospective L2 book collector for Polymarket CLOB."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
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
    "utc_date": pl.Utf8,
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
    snapshots_written: int = 0
    last_message_at: float | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    terminal_error: str | None = None
    emitting_tokens: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _GapReconnectLog:
    gap_events: list[dict[str, Any]] = field(default_factory=list)
    reconnect_events: list[dict[str, Any]] = field(default_factory=list)


def _reject_fixture_paths(*paths: Path | str) -> None:
    try:
        from polymarket_analytics.l2.session import reject_fixture_paths

        reject_fixture_paths(*paths)
    except ImportError:
        markers = ("fixtures", "books_smoke", "tests/fixtures")
        for raw in paths:
            text = str(raw).replace("\\", "/").lower()
            for marker in markers:
                if marker in text:
                    raise ValueError(f"output path looks like a fixture: {raw}")


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


def rebuild_normalized_from_raw(raw_path: Path, out_path: Path, *, session_id: str, utc_date: str | None = None) -> dict[str, Any]:
    """Rebuild normalized parquet from raw JSONL (recovery path)."""
    rows: list[dict[str, Any]] = []
    malformed = 0
    utc_date = utc_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _iter_lines() -> Any:
        if str(raw_path).endswith(".gz"):
            with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    yield line
        else:
            with raw_path.open() as handle:
                for line in handle:
                    yield line

    for line in _iter_lines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        items = item if isinstance(item, list) else [item]
        for payload in items:
            if not isinstance(payload, dict):
                malformed += 1
                continue
            book = parse_book_message(payload)
            trade = parse_trade_message(payload)
            if book:
                rows.append({**book, "session_id": session_id, "utc_date": utc_date})
            if trade:
                rows.append({**trade, "session_id": session_id, "utc_date": utc_date})
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


def _apply_fail_closed(
    report: dict[str, Any],
    *,
    health: SessionHealth,
    duration_sec: float,
) -> None:
    actual_duration = (health.ended_at or time.time()) - health.started_at
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
    elif not health.emitting_tokens:
        report["ok"] = False
        report["error"] = "no emitting tokens"
    elif actual_duration < min_ok_duration and duration_sec >= 30:
        report["ok"] = False
        report["error"] = (
            f"collection ended early: actual={actual_duration:.1f}s requested={duration_sec:.1f}s"
        )


async def _collect_ws(
    token_ids: list[str],
    *,
    duration_sec: float,
    out_raw: Path,
    out_normalized: Path,
    health: SessionHealth,
    session_id: str,
    utc_date: str,
    compress_raw: bool = False,
    snapshot_dir: Path | None = None,
    snapshot_every_sec: float = 60.0,
    snapshot_every_messages: int = 5000,
    gap_log: _GapReconnectLog | None = None,
    log_path: Path | None = None,
) -> None:
    import websockets

    out_raw.mkdir(parents=True, exist_ok=True)
    out_normalized.mkdir(parents=True, exist_ok=True)
    raw_name = f"session_{session_id}.jsonl.gz" if compress_raw else f"session_{session_id}.jsonl"
    raw_path = out_raw / raw_name
    norm_rows: list[dict[str, Any]] = []
    prev_seq: int | None = None
    flush_index = 0
    flush_every = 500
    emitting: set[str] = set()
    books: dict[str, dict[str, float | None]] = {}
    last_snapshot_at = time.time()
    msgs_since_snapshot = 0
    snapshot_index = 0

    logger = logging.getLogger("book_collector")
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def flush_normalized() -> None:
        nonlocal flush_index
        if not norm_rows:
            return
        path = out_normalized / f"session_{session_id}_{flush_index:05d}.parquet"
        write_normalized_rows(norm_rows, path)
        health.normalized_rows += len(norm_rows)
        norm_rows.clear()
        flush_index += 1

    def write_snapshot(reason: str) -> None:
        nonlocal snapshot_index, last_snapshot_at, msgs_since_snapshot
        if snapshot_dir is None:
            return
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": session_id,
            "utc_date": utc_date,
            "index": snapshot_index,
            "reason": reason,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "books": [
                {"asset_id": aid, "best_bid": lv.get("best_bid"), "best_ask": lv.get("best_ask")}
                for aid, lv in sorted(books.items())
            ],
        }
        snap_path = snapshot_dir / f"snapshot_{snapshot_index:05d}.json"
        snap_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        health.snapshots_written += 1
        snapshot_index += 1
        last_snapshot_at = time.time()
        msgs_since_snapshot = 0

    def append_raw(line: str) -> None:
        if compress_raw:
            with gzip.open(raw_path, "at", encoding="utf-8") as fh:
                fh.write(line + "\n")
        else:
            with raw_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    subscribe = {"assets_ids": token_ids, "type": "market"}
    deadline = time.time() + duration_sec

    while time.time() < deadline:
        try:
            async with websockets.connect(WS_MARKET_URL, ping_interval=20) as ws:
                health.connected = True
                await ws.send(json.dumps(subscribe))
                logger.info("connected session=%s tokens=%d", session_id, len(token_ids))
                while time.time() < deadline:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        if snapshot_dir and time.time() - last_snapshot_at >= snapshot_every_sec:
                            write_snapshot("periodic_timeout")
                        continue
                    health.messages_received += 1
                    health.last_message_at = time.time()
                    msgs_since_snapshot += 1
                    try:
                        payload = json.loads(msg)
                    except json.JSONDecodeError:
                        health.malformed_messages += 1
                        continue

                    items = payload if isinstance(payload, list) else [payload]
                    for item in items:
                        if not isinstance(item, dict):
                            health.malformed_messages += 1
                            continue
                        append_raw(json.dumps(item))
                        health.raw_rows += 1
                        seq = item.get("sequence") or item.get("seq")
                        if isinstance(seq, int) and detect_sequence_gap(prev_seq, seq):
                            health.gaps_detected += 1
                            if gap_log is not None:
                                gap_log.gap_events.append(
                                    {
                                        "prev_seq": prev_seq,
                                        "seq": seq,
                                        "at": datetime.now(timezone.utc).isoformat(),
                                    }
                                )
                        if isinstance(seq, int):
                            prev_seq = seq
                        book = parse_book_message(item)
                        trade = parse_trade_message(item)
                        if book:
                            aid = book.get("asset_id")
                            if aid:
                                emitting.add(str(aid))
                                books[str(aid)] = {
                                    "best_bid": book.get("best_bid"),
                                    "best_ask": book.get("best_ask"),
                                }
                            norm_rows.append({**book, "session_id": session_id, "utc_date": utc_date})
                        if trade:
                            aid = trade.get("asset_id")
                            if aid:
                                emitting.add(str(aid))
                            norm_rows.append({**trade, "session_id": session_id, "utc_date": utc_date})
                        if len(norm_rows) >= flush_every:
                            flush_normalized()
                    if snapshot_dir and (
                        time.time() - last_snapshot_at >= snapshot_every_sec
                        or msgs_since_snapshot >= snapshot_every_messages
                    ):
                        write_snapshot("periodic")
        except Exception as exc:  # reconnect; preserve raw already flushed
            health.reconnects += 1
            health.connected = False
            health.terminal_error = f"{type(exc).__name__}: {exc}"
            if gap_log is not None:
                gap_log.reconnect_events.append(
                    {
                        "error": health.terminal_error,
                        "at": datetime.now(timezone.utc).isoformat(),
                        "reconnect_n": health.reconnects,
                    }
                )
            logger.warning("reconnect n=%d err=%s", health.reconnects, exc)
            await asyncio.sleep(1.0)

    flush_normalized()
    if snapshot_dir and books:
        write_snapshot("final")
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
    _reject_fixture_paths(data_root)
    session_id = session_id or uuid.uuid4().hex[:12]
    run_dir = data_root / "books" / "runs" / (run_dir_name or session_id)
    _reject_fixture_paths(run_dir)
    tokens = [str(t) for t in token_ids[:max_tokens]]
    health = SessionHealth()
    utc_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
                utc_date=utc_date,
                compress_raw=False,
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
    _apply_fail_closed(report, health=health, duration_sec=duration_sec)

    health_path = run_dir / "collector_health.json"
    health_path.parent.mkdir(parents=True, exist_ok=True)
    health_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def run_session_collection(
    *,
    duration_sec: float = 18000.0,
    universe_mode: str = "diversified",
    seed: int = 42,
    n_core: int = 60,
    n_rotate: int = 40,
    n_tokens: int = 100,
    data_root: Path | str = "data",
    session_id: str | None = None,
    token_ids: list[str] | None = None,
    snapshot_every_sec: float = 60.0,
    snapshot_every_messages: int = 5000,
) -> dict[str, Any]:
    """
    Full continuous session capture with diversified universe and session layout.

    Writes under ``data/books/sessions/<session_id>/`` and validates outputs.
    """
    from polymarket_analytics.l2.session import (
        build_session_manifest,
        ensure_session_layout,
        fingerprint_session,
        reject_fixture_paths,
        utc_date_now,
        validate_session_outputs,
        write_gap_report,
        write_metadata,
        write_session_manifest,
    )
    from polymarket_analytics.l2.universe import (
        build_diversified_universe,
        build_top_universe,
        fetch_market_metadata,
    )

    data_root = Path(data_root)
    reject_fixture_paths(data_root)
    session_id = session_id or uuid.uuid4().hex[:12]
    session_dir = data_root / "books" / "sessions" / session_id
    reject_fixture_paths(session_dir)
    layout = ensure_session_layout(session_dir)
    utc_date = utc_date_now()
    started_at = datetime.now(timezone.utc)

    if token_ids:
        universe = {
            "mode": "pinned",
            "seed": seed,
            "requested_tokens": len(token_ids),
            "selected_tokens": [str(t) for t in token_ids[:n_tokens]],
        }
    elif universe_mode == "top":
        universe = build_top_universe(n_tokens)
    else:
        universe = build_diversified_universe(n_core=n_core, n_rotate=n_rotate, seed=seed)

    tokens = universe["selected_tokens"][:n_tokens]
    health = SessionHealth()
    gap_log = _GapReconnectLog()
    norm_dir = layout["normalized"] / f"utc_date={utc_date}"

    try:
        asyncio.run(
            _collect_ws(
                tokens,
                duration_sec=duration_sec,
                out_raw=layout["raw"],
                out_normalized=norm_dir,
                health=health,
                session_id=session_id,
                utc_date=utc_date,
                compress_raw=True,
                snapshot_dir=layout["snapshots"],
                snapshot_every_sec=snapshot_every_sec,
                snapshot_every_messages=snapshot_every_messages,
                gap_log=gap_log,
                log_path=layout["logs"] / "collector.log",
            )
        )
    except Exception as exc:  # noqa: BLE001
        health.terminal_error = f"{type(exc).__name__}: {exc}"
        health.ended_at = time.time()

    ended_at = datetime.now(timezone.utc)
    market_meta = fetch_market_metadata(tokens)
    write_metadata(session_dir, token_ids=tokens, market_meta=market_meta)

    actual_duration = (health.ended_at or time.time()) - health.started_at
    report = {
        "ok": True,
        "session_id": session_id,
        "session_dir": f"data/books/sessions/{session_id}",
        "utc_date": utc_date,
        "universe": universe,
        "requested_tokens": n_tokens,
        "selected_tokens": tokens,
        "emitting_tokens": health.emitting_tokens,
        "duration_sec_requested": duration_sec,
        "duration_sec_actual": actual_duration,
        "health": health.to_dict(),
    }
    _apply_fail_closed(report, health=health, duration_sec=duration_sec)

    write_gap_report(
        session_dir,
        session_id=session_id,
        health=health.to_dict(),
        gap_events=gap_log.gap_events,
        reconnect_events=gap_log.reconnect_events,
    )
    health_path = session_dir / "health.json"
    health_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    manifest = build_session_manifest(
        session_id=session_id,
        start_utc=started_at,
        end_utc=ended_at,
        duration_sec=actual_duration,
        tokens_selected=tokens,
        tokens_emitting=health.emitting_tokens,
        raw_row_count=health.raw_rows,
        normalized_row_count=health.normalized_rows,
        run_dir=session_dir,
        extra={
            "utc_date": utc_date,
            "universe_mode": universe.get("mode"),
            "seed": universe.get("seed"),
            "snapshots_written": health.snapshots_written,
        },
    )
    write_session_manifest(manifest, session_dir)
    fingerprint_session(session_dir)

    validation = validate_session_outputs(session_dir, requested_duration_sec=duration_sec)
    report["validation"] = validation
    if not validation.get("ok"):
        report["ok"] = False
        report["error"] = "; ".join(validation.get("errors") or ["validation failed"])
    health_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (session_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, default=str),
        encoding="utf-8",
    )
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
