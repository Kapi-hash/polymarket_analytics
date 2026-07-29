"""Reproducible diversified token universe for L2 collection."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from typing import Any, Final
from urllib.request import Request, urlopen

GAMMA_MARKETS_URL: Final[str] = "https://gamma-api.polymarket.com/markets"
USER_AGENT: Final[str] = "polymarket-analytics-research/0.8"


def _seed_to_int(seed: int | str) -> int:
    if isinstance(seed, int):
        return seed
    digest = hashlib.sha256(str(seed).encode()).hexdigest()
    return int(digest[:16], 16)


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _market_tokens(market: dict[str, Any]) -> list[str]:
    raw = market.get("clobTokenIds") or market.get("clob_token_ids") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = []
    return [str(t) for t in raw] if isinstance(raw, list) else []


def _extract_market_row(market: dict[str, Any], *, as_of: datetime) -> dict[str, Any] | None:
    tokens = _market_tokens(market)
    if not tokens:
        return None
    token_id = tokens[0]
    created = _parse_dt(market.get("createdAt") or market.get("created_at"))
    resolution = _parse_dt(
        market.get("endDate") or market.get("end_date") or market.get("resolutionDate")
    )
    ttr_hours = None
    if resolution is not None:
        ttr_hours = max(0.0, (resolution - as_of).total_seconds() / 3600.0)
    price = market.get("lastTradePrice") or market.get("bestBid") or market.get("bestAsk")
    try:
        prob = float(price) if price is not None else None
    except (TypeError, ValueError):
        prob = None
    volume = market.get("volume") or market.get("volumeNum") or market.get("volume24hr")
    try:
        vol = float(volume) if volume is not None else None
    except (TypeError, ValueError):
        vol = None
    spread = market.get("spread")
    try:
        spr = float(spread) if spread is not None else None
    except (TypeError, ValueError):
        spr = None
    depth = market.get("liquidity") or market.get("liquidityNum")
    try:
        dep = float(depth) if depth is not None else None
    except (TypeError, ValueError):
        dep = None
    category = str(market.get("category") or market.get("groupItemTitle") or "other")
    return {
        "token_id": token_id,
        "condition_id": str(market.get("conditionId") or market.get("condition_id") or ""),
        "outcome": str(market.get("outcome") or market.get("outcomes") or ""),
        "event_id": str(market.get("eventId") or market.get("event_id") or market.get("id") or ""),
        "category": category,
        "created_at": created.isoformat() if created else None,
        "resolution_at": resolution.isoformat() if resolution else None,
        "fee_category": category.lower(),
        "probability": prob,
        "ttr_hours": ttr_hours,
        "spread": spr,
        "volume": vol,
        "depth": dep,
        "recently_created": (as_of - created).days <= 7 if created else None,
        "rapidly_changing": market.get("oneDayPriceChange") or market.get("one_day_price_change"),
        "market_metadata": {
            k: market.get(k)
            for k in ("question", "slug", "active", "closed", "enableOrderBook")
            if k in market
        },
        "all_token_ids": tokens,
    }


def _prob_bucket(p: float | None) -> str:
    if p is None:
        return "unknown"
    if p < 0.2:
        return "low"
    if p < 0.4:
        return "mid_low"
    if p < 0.6:
        return "mid"
    if p < 0.8:
        return "mid_high"
    return "high"


def _ttr_bucket(h: float | None) -> str:
    if h is None:
        return "unknown"
    if h < 24:
        return "lt_1d"
    if h < 168:
        return "lt_1w"
    if h < 720:
        return "lt_1m"
    return "gt_1m"


def _rank_liquid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(r: dict[str, Any]) -> tuple[float, float, float]:
        vol = float(r.get("volume") or 0.0)
        dep = float(r.get("depth") or 0.0)
        spr = float(r.get("spread") if r.get("spread") is not None else 1.0)
        return (-vol, -dep, spr)

    return sorted(rows, key=key)


def build_token_universe(
    markets: list[dict[str, Any]],
    *,
    n_core: int = 60,
    n_rotate: int = 40,
    seed: int | str = 0,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Build reproducible core + rotating token universe with stratified rotation."""
    as_of = as_of or datetime.now(timezone.utc)
    rng = random.Random(_seed_to_int(seed))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for market in markets:
        row = _extract_market_row(market, as_of=as_of)
        if row is None or row["token_id"] in seen:
            continue
        seen.add(row["token_id"])
        rows.append(row)

    ranked = _rank_liquid(rows)
    core = ranked[: min(n_core, len(ranked))]

    remaining = [r for r in ranked if r["token_id"] not in {c["token_id"] for c in core}]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in remaining:
        b = (
            f"p:{_prob_bucket(r.get('probability'))}|"
            f"ttr:{_ttr_bucket(r.get('ttr_hours'))}|"
            f"cat:{r.get('category')}|"
            f"spr:{r.get('spread') is not None}|"
            f"vol:{r.get('volume') is not None}|"
            f"dep:{r.get('depth') is not None}|"
            f"new:{r.get('recently_created')}|"
            f"chg:{r.get('rapidly_changing') is not None}"
        )
        buckets.setdefault(b, []).append(r)

    rotate: list[dict[str, Any]] = []
    bucket_keys = sorted(buckets.keys())
    rng.shuffle(bucket_keys)
    idx = 0
    while len(rotate) < n_rotate and bucket_keys:
        key = bucket_keys[idx % len(bucket_keys)]
        pool = buckets[key]
        if pool:
            pick = rng.choice(pool)
            rotate.append(pick)
            pool.remove(pick)
            if not pool:
                bucket_keys.remove(key)
        idx += 1
        if idx > n_rotate * 20:
            break

    return {
        "as_of": as_of.isoformat(),
        "seed": seed,
        "n_core": n_core,
        "n_rotate": n_rotate,
        "core_tokens": core,
        "rotate_tokens": rotate,
        "selected_token_ids": [r["token_id"] for r in core + rotate],
        "metadata": {
            "total_markets_scanned": len(markets),
            "eligible_tokens": len(rows),
            "core_count": len(core),
            "rotate_count": len(rotate),
        },
    }


def discover_markets_for_universe(limit: int = 500) -> list[dict[str, Any]]:
    """Fetch Gamma markets with browser-like User-Agent."""
    req = Request(
        f"{GAMMA_MARKETS_URL}?active=true&closed=false&limit={min(max(limit, 20), 500)}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode())
    if isinstance(payload, list):
        return payload
    return payload.get("markets") or payload.get("data") or []


def build_diversified_universe(
    *,
    n_core: int = 60,
    n_rotate: int = 40,
    seed: int | str = 42,
) -> dict[str, Any]:
    """Wrapper: 60 core + 40 rotate tokens via stratified universe builder."""
    markets = discover_markets_for_universe(limit=500)
    built = build_token_universe(markets, n_core=n_core, n_rotate=n_rotate, seed=seed)
    return {
        "mode": "diversified",
        "seed": seed,
        "requested_tokens": n_core + n_rotate,
        "n_core": n_core,
        "n_rotate": n_rotate,
        "selected_tokens": built["selected_token_ids"],
        "core_tokens": [r["token_id"] for r in built["core_tokens"]],
        "rotate_tokens": [r["token_id"] for r in built["rotate_tokens"]],
        "metadata": built.get("metadata"),
        "as_of": built.get("as_of"),
    }


def build_top_universe(n_tokens: int = 100) -> dict[str, Any]:
    """Volume-ranked top-N token universe."""
    from polymarket_analytics.collectors.book_collector import discover_active_token_ids

    token_ids = discover_active_token_ids(n_tokens)
    return {
        "mode": "top",
        "seed": None,
        "requested_tokens": n_tokens,
        "selected_tokens": token_ids[:n_tokens],
        "core_tokens": token_ids[:n_tokens],
        "rotate_tokens": [],
    }


def fetch_market_metadata(token_ids: list[str], *, limit: int = 200) -> list[dict[str, Any]]:
    """Best-effort Gamma metadata rows for subscribed tokens."""
    wanted = set(str(t) for t in token_ids)
    rows = discover_markets_for_universe(limit=limit)
    out: list[dict[str, Any]] = []
    for market in rows:
        toks = set(_market_tokens(market))
        if not toks & wanted:
            continue
        row = _extract_market_row(market, as_of=datetime.now(timezone.utc))
        if row is None:
            continue
        row["subscribed_token_ids"] = sorted(toks & wanted)
        out.append(row)
    return out
