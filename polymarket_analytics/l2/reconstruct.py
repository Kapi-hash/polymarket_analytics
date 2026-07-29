"""Stateful L2 book reconstruction from snapshots and incremental updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SideName = Literal["bid", "ask"]


@dataclass
class BookSide:
    """One side of the book: price -> size."""

    levels: dict[float, float] = field(default_factory=dict)

    def set_level(self, price: float, size: float) -> None:
        if size <= 0:
            self.levels.pop(float(price), None)
        else:
            self.levels[float(price)] = float(size)

    def best_price(self, *, side: SideName) -> float | None:
        if not self.levels:
            return None
        prices = sorted(self.levels.keys())
        return prices[-1] if side == "bid" else prices[0]

    def depth(self, n_levels: int = 5) -> float:
        prices = sorted(self.levels.keys(), reverse=True)
        return sum(self.levels[p] for p in prices[:n_levels])

    def sorted_levels(self, *, side: SideName) -> list[tuple[float, float]]:
        prices = sorted(self.levels.keys(), reverse=(side == "bid"))
        return [(p, self.levels[p]) for p in prices if self.levels[p] > 0]


def microprice(bid_p: float, bid_s: float, ask_p: float, ask_s: float) -> float | None:
    """Size-weighted microprice at touch."""
    den = float(bid_s) + float(ask_s)
    if den <= 0 or bid_p <= 0 or ask_p <= 0:
        return None
    return (float(ask_p) * float(bid_s) + float(bid_p) * float(ask_s)) / den


@dataclass
class OrderBook:
    """Per-asset L2 book state."""

    asset_id: str
    bids: BookSide = field(default_factory=BookSide)
    asks: BookSide = field(default_factory=BookSide)
    uncertain: bool = False
    last_seq: int | None = None
    last_exchange_time: float | None = None
    last_receive_time: float | None = None
    gaps: int = 0
    inversions: int = 0
    crossed_count: int = 0

    def apply_snapshot(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        *,
        seq: int | None = None,
        exchange_time: float | None = None,
        receive_time: float | None = None,
    ) -> None:
        self.bids = BookSide()
        self.asks = BookSide()
        for p, s in bids:
            self.bids.set_level(p, s)
        for p, s in asks:
            self.asks.set_level(p, s)
        self.uncertain = False
        if seq is not None:
            self.last_seq = seq
        self._update_times(exchange_time, receive_time)

    def apply_delta(
        self,
        side: SideName,
        price: float,
        size: float,
        *,
        seq: int | None = None,
        exchange_time: float | None = None,
        receive_time: float | None = None,
    ) -> None:
        if self.uncertain:
            return
        book = self.bids if side == "bid" else self.asks
        book.set_level(price, size)
        if seq is not None:
            if self.last_seq is not None and seq - self.last_seq > 1:
                self.gaps += 1
                self.uncertain = True
            self.last_seq = seq
        self._update_times(exchange_time, receive_time)

    def _update_times(self, exchange_time: float | None, receive_time: float | None) -> None:
        if exchange_time is not None and self.last_exchange_time is not None:
            if exchange_time < self.last_exchange_time:
                self.inversions += 1
        if exchange_time is not None:
            self.last_exchange_time = exchange_time
        if receive_time is not None:
            self.last_receive_time = receive_time

    def mark_uncertain(self) -> None:
        self.uncertain = True

    def best_bid(self) -> float | None:
        return self.bids.best_price(side="bid")

    def best_ask(self) -> float | None:
        return self.asks.best_price(side="ask")

    def mid(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def spread(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return ba - bb

    def is_crossed(self) -> bool:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return False
        return bb >= ba

    def snapshot_record(
        self,
        *,
        exchange_time: float | None = None,
        receive_time: float | None = None,
        processing_time: float | None = None,
        event_type: str = "book",
    ) -> "BookSnapshotRecord":
        bb, ba = self.best_bid(), self.best_ask()
        bid_sz = self.bids.levels.get(bb, 0.0) if bb is not None else 0.0
        ask_sz = self.asks.levels.get(ba, 0.0) if ba is not None else 0.0
        crossed = self.is_crossed()
        if crossed:
            self.crossed_count += 1
        latency = None
        if exchange_time is not None and receive_time is not None:
            latency = receive_time - exchange_time
        return BookSnapshotRecord(
            asset_id=self.asset_id,
            best_bid=bb,
            best_ask=ba,
            mid=self.mid(),
            spread=self.spread(),
            microprice=microprice(bb or 0, bid_sz, ba or 0, ask_sz) if bb and ba else None,
            bid_depth=self.bids.depth(),
            ask_depth=self.asks.depth(),
            uncertain=self.uncertain,
            crossed=crossed,
            exchange_time=exchange_time,
            receive_time=receive_time,
            processing_time=processing_time,
            latency_ms=latency * 1000.0 if latency is not None else None,
            event_type=event_type,
            bid_levels=self.bids.sorted_levels(side="bid")[:5],
            ask_levels=self.asks.sorted_levels(side="ask")[:5],
        )


@dataclass
class BookSnapshotRecord:
    asset_id: str
    best_bid: float | None
    best_ask: float | None
    mid: float | None
    spread: float | None
    microprice: float | None
    bid_depth: float
    ask_depth: float
    uncertain: bool
    crossed: bool
    exchange_time: float | None
    receive_time: float | None
    processing_time: float | None
    latency_ms: float | None
    event_type: str
    bid_levels: list[tuple[float, float]]
    ask_levels: list[tuple[float, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_levels(raw: Any) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict):
            out.append((float(item["price"]), float(item.get("size", item.get("amount", 0)))))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((float(item[0]), float(item[1])))
    return out


def apply_messages(messages: list[dict[str, Any]]) -> list[BookSnapshotRecord]:
    """Apply messages in event-time order; return snapshot records after each applied event."""
    books: dict[str, OrderBook] = {}
    records: list[BookSnapshotRecord] = []

    def _book(asset_id: str) -> OrderBook:
        if asset_id not in books:
            books[asset_id] = OrderBook(asset_id=asset_id)
        return books[asset_id]

    sorted_msgs = sorted(
        messages,
        key=lambda m: (
            float(m.get("exchange_time") or m.get("timestamp") or 0),
            int(m.get("seq") or m.get("sequence") or 0),
        ),
    )

    for msg in sorted_msgs:
        asset_id = str(msg.get("asset_id") or msg.get("token_id") or "")
        if not asset_id:
            continue
        book = _book(asset_id)
        et = msg.get("exchange_time")
        if et is None and msg.get("timestamp") is not None:
            et = float(msg["timestamp"])
        rt = msg.get("receive_time")
        if rt is None and msg.get("received_at") is not None:
            rt = float(msg["received_at"])
        pt = msg.get("processing_time")
        seq = msg.get("seq") or msg.get("sequence")
        if isinstance(seq, str) and seq.isdigit():
            seq = int(seq)

        event = msg.get("event_type") or msg.get("type") or msg.get("record_type")
        if event in {"reconnect", "gap"}:
            book.mark_uncertain()
            continue

        if event in {"snapshot", "book"} or msg.get("bids") or msg.get("asks"):
            bids = _parse_levels(msg.get("bids") or msg.get("buys"))
            asks = _parse_levels(msg.get("asks") or msg.get("sells"))
            if bids or asks:
                book.apply_snapshot(
                    bids,
                    asks,
                    seq=int(seq) if seq is not None else None,
                    exchange_time=float(et) if et is not None else None,
                    receive_time=float(rt) if rt is not None else None,
                )
                records.append(
                    book.snapshot_record(
                        exchange_time=float(et) if et is not None else None,
                        receive_time=float(rt) if rt is not None else None,
                        processing_time=float(pt) if pt is not None else None,
                        event_type="snapshot",
                    )
                )
            continue

        side_raw = msg.get("side")
        price = msg.get("price")
        size = msg.get("size")
        if side_raw and price is not None and size is not None:
            side: SideName = "bid" if str(side_raw).lower() in {"bid", "buy", "b"} else "ask"
            book.apply_delta(
                side,
                float(price),
                float(size),
                seq=int(seq) if seq is not None else None,
                exchange_time=float(et) if et is not None else None,
                receive_time=float(rt) if rt is not None else None,
            )
            records.append(
                book.snapshot_record(
                    exchange_time=float(et) if et is not None else None,
                    receive_time=float(rt) if rt is not None else None,
                    processing_time=float(pt) if pt is not None else None,
                    event_type="delta",
                )
            )

    return records
