"""Phase 4: async Polymarket CLOB market WebSocket feed + live feature windows."""

from __future__ import annotations

import asyncio
import json
import statistics
import time
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Deque, Mapping, Sequence

MARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
PING_INTERVAL_SEC = 10.0
DEFAULT_WINDOW_SEC = 3600.0  # 1h
DEFAULT_LOOKBACK_24H_SEC = 86400.0


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return x


def _parse_ts_ms(value: Any) -> float:
    """Return unix seconds from ms/s/ISO-ish payload timestamps."""
    if value is None:
        return time.time()
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            return ts / 1000.0
        if ts > 1e10:
            return ts / 1000.0
        return ts
    text = str(value).strip()
    try:
        ts = float(text)
        if ts > 1e12:
            return ts / 1000.0
        if ts > 1e10:
            return ts / 1000.0
        return ts
    except ValueError:
        return time.time()


@dataclass(frozen=True)
class LiveTradeTick:
    token_id: str
    condition_id: str
    price: float
    size: float
    side: str
    ts: float  # unix seconds
    event_type: str = "last_trade_price"


@dataclass
class TopOfBook:
    token_id: str
    condition_id: str
    best_bid: float | None = None
    best_ask: float | None = None
    ts: float = 0.0

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return 0.5 * (self.best_bid + self.best_ask)


@dataclass
class LiveFeatures:
    token_id: str
    condition_id: str
    price: float
    size: float
    ts: float
    momentum_1h: float | None
    whale_ratio: float | None
    n_trades_1h: int
    best_bid: float | None = None
    best_ask: float | None = None


@dataclass
class TokenWindow:
    """Sliding trade window for one token_id."""

    trades: Deque[tuple[float, float, float]] = field(default_factory=deque)
    # (ts, price, size)

    def add(self, ts: float, price: float, size: float, *, max_age_sec: float) -> None:
        self.trades.append((ts, price, size))
        self._evict(ts, max_age_sec)

    def _evict(self, now: float, max_age_sec: float) -> None:
        cutoff = now - max_age_sec
        while self.trades and self.trades[0][0] < cutoff:
            self.trades.popleft()

    def features(
        self,
        *,
        now: float,
        window_1h: float = DEFAULT_WINDOW_SEC,
        window_24h: float = DEFAULT_LOOKBACK_24H_SEC,
    ) -> tuple[float | None, float | None, int]:
        """Return (momentum_1h, whale_ratio, n_trades_1h)."""
        self._evict(now, window_24h)
        if not self.trades:
            return None, None, 0

        recent = [t for t in self.trades if t[0] >= now - window_1h]
        if not recent:
            return None, None, 0

        p0 = recent[0][1]
        p1 = recent[-1][1]
        dt_h = max((recent[-1][0] - recent[0][0]) / 3600.0, 1e-6)
        momentum = (p1 - p0) / dt_h

        sizes_1h = [t[2] for t in recent if t[2] > 0]
        sizes_24h = [t[2] for t in self.trades if t[2] > 0]
        whale = None
        if sizes_1h and sizes_24h:
            med = statistics.median(sizes_24h)
            if med > 0:
                whale = (sum(sizes_1h) / len(sizes_1h)) / med

        return momentum, whale, len(recent)


class LiveFeatureEngine:
    """In-memory rolling feature engine over live trade ticks."""

    def __init__(
        self,
        *,
        window_1h_sec: float = DEFAULT_WINDOW_SEC,
        window_24h_sec: float = DEFAULT_LOOKBACK_24H_SEC,
    ) -> None:
        self.window_1h_sec = window_1h_sec
        self.window_24h_sec = window_24h_sec
        self._windows: dict[str, TokenWindow] = defaultdict(TokenWindow)
        self._books: dict[str, TopOfBook] = {}
        self._condition: dict[str, str] = {}

    def on_trade(self, tick: LiveTradeTick) -> LiveFeatures:
        self._condition[tick.token_id] = tick.condition_id
        win = self._windows[tick.token_id]
        win.add(tick.ts, tick.price, tick.size, max_age_sec=self.window_24h_sec)
        mom, whale, n = win.features(
            now=tick.ts,
            window_1h=self.window_1h_sec,
            window_24h=self.window_24h_sec,
        )
        book = self._books.get(tick.token_id)
        return LiveFeatures(
            token_id=tick.token_id,
            condition_id=tick.condition_id,
            price=tick.price,
            size=tick.size,
            ts=tick.ts,
            momentum_1h=mom,
            whale_ratio=whale,
            n_trades_1h=n,
            best_bid=book.best_bid if book else None,
            best_ask=book.best_ask if book else None,
        )

    def on_book(self, book: TopOfBook) -> None:
        self._books[book.token_id] = book
        self._condition[book.token_id] = book.condition_id


def parse_market_event(payload: Mapping[str, Any]) -> list[LiveTradeTick | TopOfBook]:
    """Normalize a WS market payload into ticks / top-of-book updates."""
    out: list[LiveTradeTick | TopOfBook] = []
    # Some frames are lists of events
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping):
                out.extend(parse_market_event(item))
        return out

    event_type = str(payload.get("event_type") or payload.get("type") or "").lower()
    asset_id = str(payload.get("asset_id") or "").strip()
    market = str(payload.get("market") or payload.get("condition_id") or "").strip()
    ts = _parse_ts_ms(payload.get("timestamp"))

    if event_type in {"last_trade_price", "last_trade", "trade"}:
        price = _finite(payload.get("price"))
        size = _finite(payload.get("size") or payload.get("amount") or 0.0) or 0.0
        side = str(payload.get("side") or "BUY").upper()
        if asset_id and price is not None and 0.0 <= price <= 1.0:
            out.append(
                LiveTradeTick(
                    token_id=asset_id,
                    condition_id=market,
                    price=price,
                    size=max(size, 0.0),
                    side=side,
                    ts=ts,
                    event_type=event_type,
                )
            )
        return out

    if event_type in {"best_bid_ask", "book"}:
        if event_type == "best_bid_ask":
            bid = _finite(payload.get("best_bid"))
            ask = _finite(payload.get("best_ask"))
        else:
            bids = payload.get("bids") or []
            asks = payload.get("asks") or []
            bid = None
            ask = None
            if isinstance(bids, list) and bids:
                # CLOB books often sorted ascending; best bid = max price
                prices = [_finite(x.get("price")) for x in bids if isinstance(x, Mapping)]
                prices = [p for p in prices if p is not None]
                bid = max(prices) if prices else None
            if isinstance(asks, list) and asks:
                prices = [_finite(x.get("price")) for x in asks if isinstance(x, Mapping)]
                prices = [p for p in prices if p is not None]
                ask = min(prices) if prices else None
        if asset_id:
            out.append(
                TopOfBook(
                    token_id=asset_id,
                    condition_id=market,
                    best_bid=bid,
                    best_ask=ask,
                    ts=ts,
                )
            )
        return out

    # price_change may include best_bid / best_ask nested
    if event_type == "price_change":
        changes = payload.get("price_changes") or []
        if isinstance(changes, list):
            for ch in changes:
                if not isinstance(ch, Mapping):
                    continue
                tid = str(ch.get("asset_id") or asset_id).strip()
                if not tid:
                    continue
                out.append(
                    TopOfBook(
                        token_id=tid,
                        condition_id=market,
                        best_bid=_finite(ch.get("best_bid")),
                        best_ask=_finite(ch.get("best_ask")),
                        ts=ts,
                    )
                )
    return out


def fetch_active_token_ids(*, limit: int = 20, timeout_sec: float = 15.0) -> list[str]:
    """Pull a small set of active Gamma market token IDs for subscription."""
    url = f"{GAMMA_MARKETS_URL}?active=true&closed=false&limit={int(limit)}"
    req = urllib.request.Request(url, headers={"User-Agent": "polymarket-analytics/0.4"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    if not isinstance(raw, list):
        return []

    token_ids: list[str] = []
    for m in raw:
        if not isinstance(m, Mapping):
            continue
        clob = m.get("clobTokenIds") or m.get("clob_token_ids")
        tokens: list[str] = []
        if isinstance(clob, str):
            try:
                parsed = json.loads(clob)
                if isinstance(parsed, list):
                    tokens = [str(x) for x in parsed]
            except json.JSONDecodeError:
                tokens = []
        elif isinstance(clob, list):
            tokens = [str(x) for x in clob]
        for t in tokens:
            if t and t not in token_ids:
                token_ids.append(t)
            if len(token_ids) >= limit:
                return token_ids
    return token_ids


class MarketWebSocketFeed:
    """Async client for Polymarket CLOB market channel (paper / research only)."""

    def __init__(
        self,
        token_ids: Sequence[str],
        *,
        url: str = MARKET_WS_URL,
        ping_interval_sec: float = PING_INTERVAL_SEC,
        feature_engine: LiveFeatureEngine | None = None,
        on_features: Callable[[LiveFeatures], None] | None = None,
    ) -> None:
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        self.token_ids = [str(t) for t in token_ids]
        self.url = url
        self.ping_interval_sec = ping_interval_sec
        self.engine = feature_engine or LiveFeatureEngine()
        self.on_features = on_features
        self._stop = asyncio.Event()
        self.events_received = 0
        self.trades_received = 0
        self.last_error: str | None = None

    def stop(self) -> None:
        self._stop.set()

    def _subscribe_payload(self) -> str:
        return json.dumps(
            {
                "type": "market",
                "assets_ids": self.token_ids,
                "custom_feature_enabled": True,
            }
        )

    async def _ping_loop(self, ws: Any) -> None:
        while not self._stop.is_set():
            try:
                await ws.send("PING")
            except Exception:
                return
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.ping_interval_sec)
                return
            except asyncio.TimeoutError:
                continue

    def handle_message(self, raw: str | bytes) -> list[LiveFeatures]:
        """Parse one WS message; update engine; return emitted feature snapshots."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        text = str(raw).strip()
        if not text or text.upper() in {"PONG", "PING"}:
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []

        features: list[LiveFeatures] = []
        for item in parse_market_event(payload):
            self.events_received += 1
            if isinstance(item, TopOfBook):
                self.engine.on_book(item)
            elif isinstance(item, LiveTradeTick):
                self.trades_received += 1
                feat = self.engine.on_trade(item)
                features.append(feat)
                if self.on_features is not None:
                    self.on_features(feat)
        return features

    async def stream(self) -> AsyncIterator[LiveFeatures]:
        """Connect, subscribe, and yield live feature snapshots from trades.

        Automatically reconnects with backoff until ``stop()`` is called.
        """
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError(
                "websockets package required: pip3 install websockets"
            ) from exc

        queue: asyncio.Queue[LiveFeatures] = asyncio.Queue()
        prev = self.on_features

        def _enqueue(feat: LiveFeatures) -> None:
            queue.put_nowait(feat)
            if prev is not None:
                prev(feat)

        self.on_features = _enqueue
        backoff = 1.0

        try:
            while not self._stop.is_set():
                try:
                    async with websockets.connect(
                        self.url,
                        ping_interval=None,
                        max_size=8_000_000,
                        open_timeout=20,
                        close_timeout=5,
                    ) as ws:
                        await ws.send(self._subscribe_payload())
                        backoff = 1.0
                        self.last_error = None
                        ping_task = asyncio.create_task(self._ping_loop(ws))
                        try:
                            while not self._stop.is_set():
                                recv = asyncio.create_task(ws.recv())
                                stop = asyncio.create_task(self._stop.wait())
                                done, pending = await asyncio.wait(
                                    {recv, stop},
                                    return_when=asyncio.FIRST_COMPLETED,
                                )
                                for t in pending:
                                    t.cancel()
                                if stop in done:
                                    return
                                try:
                                    message = recv.result()
                                except Exception as exc:
                                    self.last_error = str(exc)
                                    break
                                self.handle_message(message)
                                while not queue.empty():
                                    yield queue.get_nowait()
                        finally:
                            ping_task.cancel()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.last_error = str(exc)

                if self._stop.is_set():
                    return
                # Brief reconnect pause; keep draining any queued features
                while not queue.empty():
                    yield queue.get_nowait()
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                    return
                except asyncio.TimeoutError:
                    backoff = min(backoff * 2.0, 30.0)
        finally:
            self.on_features = prev

    async def run_until(
        self,
        *,
        duration_sec: float,
        consumer: Callable[[LiveFeatures], None] | None = None,
    ) -> int:
        """Run the feed for ``duration_sec`` seconds; return trade count."""
        end = time.time() + duration_sec
        count = 0
        async for feat in self.stream():
            count += 1
            if consumer is not None:
                consumer(feat)
            if time.time() >= end or self._stop.is_set():
                self.stop()
                break
        return count


def inject_demo_ticks(
    engine: LiveFeatureEngine,
    *,
    token_id: str = "demo_token",
    condition_id: str = "demo_condition",
    n: int = 30,
    base_price: float = 0.35,
) -> list[LiveFeatures]:
    """Offline synthetic ticks for dashboard/tests without a live socket.

    Seeds many small prints across ~24h, then a 1h burst of large sizes so
    whale_ratio exceeds 3. Price ramps within 0.30–0.40 with positive momentum
    to match the validated OOS mid/low-bucket whale setups.
    """
    now = time.time()
    out: list[LiveFeatures] = []
    # Baseline: small trades over ~20h (median size stays ~10)
    n_base = max(n * 2, 40)
    for i in range(n_base):
        ts = now - (20 * 3600) + i * ((20 * 3600) / n_base)
        tick = LiveTradeTick(
            token_id=token_id,
            condition_id=condition_id,
            price=max(base_price - 0.04, 0.05),
            size=10.0,
            side="BUY",
            ts=ts,
        )
        engine.on_trade(tick)

    # Recent burst: large sizes, rising price inside 0.30–0.40
    for i in range(n):
        frac = i / max(n - 1, 1)
        price = min(max(base_price - 0.02 + 0.04 * frac, 0.01), 0.99)
        tick = LiveTradeTick(
            token_id=token_id,
            condition_id=condition_id,
            price=price,
            size=80.0,
            side="BUY",
            ts=now - (n - i) * 30.0,
        )
        out.append(engine.on_trade(tick))
    return out
