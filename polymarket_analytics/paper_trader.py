"""Phase 4/5: paper trading — dynamic taker fees, resolution, TP/SL (no real orders)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from polymarket_analytics.backtest import StrategyParams, params_from_edge_row
from polymarket_analytics.live_feed import LiveFeatures
from polymarket_analytics.schema import PRICE_BUCKET_BREAKS, PRICE_BUCKET_LABELS

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"

# Official Polymarket category taker feeRate inputs (docs.polymarket.com/trading/fees).
# fee = C × feeRate × p × (1 − p); peaks at p = 0.50.
CATEGORY_FEE_RATES: dict[str, float] = {
    "crypto": 0.07,
    "sports": 0.05,
    "finance": 0.04,
    "politics": 0.04,
    "economics": 0.05,
    "culture": 0.05,
    "weather": 0.05,
    "other": 0.05,
    "general": 0.05,
    "mentions": 0.04,
    "tech": 0.04,
    "geopolitics": 0.0,
}

# Conservative default when OOS report is missing
DEFAULT_PAPER_STRATEGIES: tuple[StrategyParams, ...] = (
    StrategyParams(
        price_bucket="0.40-0.50",
        min_whale_ratio=3.0,
        momentum_1h="any",
        side="BUY",
    ),
    StrategyParams(
        price_bucket="0.30-0.40",
        min_whale_ratio=3.0,
        momentum_1h="pos",
        side="BUY",
    ),
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def price_bucket_label(price: float) -> str:
    """Map a mid/last price into the same bucket labels as Phase 2."""
    breaks = PRICE_BUCKET_BREAKS
    labels = PRICE_BUCKET_LABELS
    for i, b in enumerate(breaks):
        if price < b:
            return labels[i]
    return labels[-1]


def resolve_fee_rate(
    category: str | None = None,
    fee_rate: float | None = None,
) -> float:
    """Return the taker feeRate constant for the dynamic fee curve."""
    if fee_rate is not None:
        return max(0.0, float(fee_rate))
    key = (category or "crypto").strip().lower()
    return float(CATEGORY_FEE_RATES.get(key, CATEGORY_FEE_RATES["crypto"]))


def dynamic_taker_fee(
    shares: float,
    price: float,
    *,
    fee_rate: float,
) -> float:
    """
    Polymarket dynamic taker fee (USDC):

        fee = C × feeRate × p × (1 − p)

    Peaks at p = 0.50 (parabolic in p).
    """
    c = max(float(shares), 0.0)
    p = min(max(float(price), 0.0), 1.0)
    r = max(float(fee_rate), 0.0)
    return c * r * p * (1.0 - p)


def relative_taker_fee_rate(price: float, *, fee_rate: float) -> float:
    """Fee as a fraction of notional (= fee / (C·p) = feeRate · (1 − p))."""
    p = min(max(float(price), 1e-12), 1.0)
    return max(float(fee_rate), 0.0) * (1.0 - p)


def kelly_fraction(
    win_rate: float,
    *,
    payout_odds: float = 1.0,
    fraction: float = 0.25,
    max_fraction: float = 0.05,
) -> float:
    """Fractional Kelly for binary contracts."""
    p = max(0.0, min(1.0, win_rate))
    b = max(payout_odds, 1e-9)
    q = 1.0 - p
    full = (b * p - q) / b
    if full <= 0:
        return 0.0
    return min(full * fraction, max_fraction)


@dataclass
class PaperConfig:
    bankroll: float = 10_000.0
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.05
    # Dynamic taker fees (preferred). Legacy flat fee_bps kept for override.
    fee_category: str = "crypto"
    fee_rate: float | None = None
    use_dynamic_fees: bool = True
    fee_bps: float = 0.0  # if use_dynamic_fees=False: flat bps of notional
    spread_slippage_bps: float = 50.0
    min_oos_ev_pct: float = 10.0
    require_persists: bool = True
    max_open_positions: int = 25
    cooldown_sec: float = 60.0
    # Optional early exits (fractions of entry fill price)
    take_profit_pct: float | None = None
    stop_loss_pct: float | None = None
    resolve_poll_sec: float = 60.0

    def effective_fee_rate(self) -> float:
        return resolve_fee_rate(self.fee_category, self.fee_rate)


@dataclass
class PaperPosition:
    position_id: str
    token_id: str
    condition_id: str
    strategy_label: str
    entry_ts: float
    entry_price_raw: float
    entry_price_fill: float
    shares: float
    notional: float
    fees: float
    mark_price: float
    fee_rate: float = 0.0
    entry_ev_pct_gross: float = 0.0
    entry_ev_pct_net: float = 0.0
    status: str = "open"  # open | resolved | take_profit | stop_loss
    exit_reason: str | None = None
    exit_ts: float | None = None
    exit_price: float | None = None
    exit_fees: float = 0.0
    realized_pnl: float | None = None
    win: bool | None = None

    @property
    def unrealized_pnl(self) -> float:
        if self.status != "open":
            return 0.0
        # Mark-to-market vs full entry cost (notional + taker fee already paid)
        return self.mark_price * self.shares - self.notional - self.fees

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["unrealized_pnl"] = self.unrealized_pnl
        return d


@dataclass
class SignalEvent:
    ts: float
    token_id: str
    condition_id: str
    price: float
    whale_ratio: float | None
    momentum_1h: float | None
    price_bucket: str
    strategy_label: str
    oos_ev_pct: float
    oos_win_rate: float
    fee_estimate: float = 0.0
    net_ev_pct: float = 0.0


@dataclass
class PaperTrader:
    """Evaluate live features vs OOS strategies; simulate fills only."""

    config: PaperConfig = field(default_factory=PaperConfig)
    strategies: list[tuple[StrategyParams, float, float]] = field(default_factory=list)
    positions: list[PaperPosition] = field(default_factory=list)
    signals: list[SignalEvent] = field(default_factory=list)
    journal_path: Path | None = None
    cash: float = 0.0
    realized_pnl: float = 0.0
    ticks_seen: int = 0
    signals_fired: int = 0
    fills: int = 0
    resolutions: int = 0
    _last_entry_ts: dict[str, float] = field(default_factory=dict)
    _last_resolve_poll: float = 0.0

    def __post_init__(self) -> None:
        if self.cash <= 0:
            self.cash = float(self.config.bankroll)

    @classmethod
    def from_oos_report(
        cls,
        report_path: Path | str | None,
        *,
        config: PaperConfig | None = None,
        journal_path: Path | str | None = None,
    ) -> "PaperTrader":
        cfg = config or PaperConfig()
        strategies = load_oos_strategies(
            report_path,
            min_oos_ev_pct=cfg.min_oos_ev_pct,
            require_persists=cfg.require_persists,
        )
        return cls(
            config=cfg,
            strategies=strategies,
            journal_path=Path(journal_path) if journal_path else None,
        )

    @property
    def open_positions(self) -> list[PaperPosition]:
        return [p for p in self.positions if p.status == "open"]

    @property
    def equity(self) -> float:
        return self.cash + sum(p.mark_price * p.shares for p in self.open_positions)

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.open_positions)

    def compute_taker_fee(self, shares: float, price: float) -> float:
        if not self.config.use_dynamic_fees:
            notional = max(shares, 0.0) * max(price, 0.0)
            return notional * (self.config.fee_bps / 10_000.0)
        return dynamic_taker_fee(
            shares, price, fee_rate=self.config.effective_fee_rate()
        )

    def matches(self, feat: LiveFeatures, params: StrategyParams) -> bool:
        bucket = price_bucket_label(feat.price)
        if params.price_bucket is not None and bucket != params.price_bucket:
            return False
        if params.min_whale_ratio is not None:
            if feat.whale_ratio is None or feat.whale_ratio <= params.min_whale_ratio:
                return False
        if params.momentum_1h == "pos":
            if feat.momentum_1h is None or feat.momentum_1h <= 0:
                return False
        elif params.momentum_1h == "neg":
            if feat.momentum_1h is None or feat.momentum_1h >= 0:
                return False
        if params.min_volume_spike is not None:
            return False
        if params.max_time_to_resolution_hours is not None:
            return False
        if params.require_price_volume_divergence:
            if feat.momentum_1h is None or feat.momentum_1h >= 0:
                return False
            if feat.whale_ratio is None or feat.whale_ratio <= 3.0:
                return False
        return True

    def apply_fill_price(self, raw_price: float, *, best_ask: float | None) -> float:
        """Model buy slippage only (fees applied separately via dynamic curve)."""
        slip = self.config.spread_slippage_bps / 10_000.0
        base = best_ask if best_ask is not None else raw_price
        fill = min(max(base * (1.0 + slip), 0.01), 0.99)
        return fill

    def _size_buy(self, budget: float, fill_px: float) -> tuple[float, float, float]:
        """
        Size shares so notional + dynamic fee ≤ budget.

        cost = C·p + C·feeRate·p·(1−p) = C·p·(1 + feeRate·(1−p))
        """
        p = max(fill_px, 1e-9)
        if self.config.use_dynamic_fees:
            r = self.config.effective_fee_rate()
            denom = p * (1.0 + r * (1.0 - p))
        else:
            denom = p * (1.0 + self.config.fee_bps / 10_000.0)
        shares = budget / max(denom, 1e-12)
        notional = shares * p
        fees = self.compute_taker_fee(shares, p)
        return shares, notional, fees

    def on_features(self, feat: LiveFeatures) -> PaperPosition | None:
        """Mark open positions, run TP/SL + resolution polls, maybe open a fill."""
        self.ticks_seen += 1
        for pos in self.open_positions:
            if pos.token_id == feat.token_id:
                pos.mark_price = feat.price

        # Early exits before new entries
        self._evaluate_stops(feat.ts)
        self.maybe_poll_resolutions(now=feat.ts)

        if len(self.open_positions) >= self.config.max_open_positions:
            return None
        last = self._last_entry_ts.get(feat.token_id, 0.0)
        if feat.ts - last < self.config.cooldown_sec:
            return None
        if any(p.token_id == feat.token_id for p in self.open_positions):
            return None

        fee_rate = self.config.effective_fee_rate()
        for params, oos_ev, oos_wr in self.strategies:
            if not self.matches(feat, params):
                continue

            fill_px = self.apply_fill_price(feat.price, best_ask=feat.best_ask)
            rel_fee = relative_taker_fee_rate(fill_px, fee_rate=fee_rate)
            if not self.config.use_dynamic_fees:
                rel_fee = self.config.fee_bps / 10_000.0
            net_ev = float(oos_ev) - 100.0 * rel_fee

            self.signals_fired += 1
            sig = SignalEvent(
                ts=feat.ts,
                token_id=feat.token_id,
                condition_id=feat.condition_id,
                price=feat.price,
                whale_ratio=feat.whale_ratio,
                momentum_1h=feat.momentum_1h,
                price_bucket=price_bucket_label(feat.price),
                strategy_label=params.label(),
                oos_ev_pct=oos_ev,
                oos_win_rate=oos_wr,
                fee_estimate=rel_fee,
                net_ev_pct=net_ev,
            )
            self.signals.append(sig)

            # Skip if fees wipe the gross OOS edge
            if net_ev < 0:
                continue

            b = (1.0 - feat.price) / max(feat.price, 1e-9)
            f = kelly_fraction(
                oos_wr,
                payout_odds=b,
                fraction=self.config.kelly_fraction,
                max_fraction=self.config.max_position_pct,
            )
            if f <= 0:
                continue
            budget = self.equity * f
            if budget < 1.0 or budget > self.cash:
                continue

            shares, notional, fees = self._size_buy(budget, fill_px)
            cost = notional + fees
            if cost > self.cash or shares <= 0:
                continue

            self.cash -= cost
            pos = PaperPosition(
                position_id=str(uuid.uuid4())[:8],
                token_id=feat.token_id,
                condition_id=feat.condition_id,
                strategy_label=params.label(),
                entry_ts=feat.ts,
                entry_price_raw=feat.price,
                entry_price_fill=fill_px,
                shares=shares,
                notional=notional,
                fees=fees,
                mark_price=feat.price,
                fee_rate=fee_rate if self.config.use_dynamic_fees else 0.0,
                entry_ev_pct_gross=float(oos_ev),
                entry_ev_pct_net=net_ev,
            )
            self.positions.append(pos)
            self._last_entry_ts[feat.token_id] = feat.ts
            self.fills += 1
            self.persist()
            return pos
        return None

    def _evaluate_stops(self, now: float) -> None:
        tp = self.config.take_profit_pct
        sl = self.config.stop_loss_pct
        if tp is None and sl is None:
            return
        for pos in list(self.open_positions):
            if pos.entry_price_fill <= 0:
                continue
            ret = (pos.mark_price - pos.entry_price_fill) / pos.entry_price_fill
            if tp is not None and ret >= tp:
                self.close_position(
                    pos.position_id,
                    exit_price=pos.mark_price,
                    exit_ts=now,
                    reason="take_profit",
                )
            elif sl is not None and ret <= -abs(sl):
                self.close_position(
                    pos.position_id,
                    exit_price=pos.mark_price,
                    exit_ts=now,
                    reason="stop_loss",
                )

    def close_position(
        self,
        position_id: str,
        *,
        exit_price: float,
        exit_ts: float | None = None,
        reason: str = "manual",
        apply_exit_fee: bool = True,
    ) -> PaperPosition | None:
        """Market exit prior to resolution (TP/SL); applies taker fee on sell."""
        for pos in self.positions:
            if pos.position_id != position_id or pos.status != "open":
                continue
            px = min(max(float(exit_price), 0.0), 1.0)
            exit_fees = (
                self.compute_taker_fee(pos.shares, px) if apply_exit_fee else 0.0
            )
            proceeds = max(px * pos.shares - exit_fees, 0.0)
            pnl = proceeds - pos.notional - pos.fees
            self.cash += proceeds
            self.realized_pnl += pnl
            pos.status = reason if reason in {"take_profit", "stop_loss"} else "resolved"
            pos.exit_reason = reason
            pos.exit_ts = exit_ts if exit_ts is not None else time.time()
            pos.exit_price = px
            pos.exit_fees = exit_fees
            pos.realized_pnl = pnl
            pos.win = pnl > 0
            pos.mark_price = px
            self.persist()
            return pos
        return None

    def resolve_position(
        self,
        position_id: str,
        *,
        token_won: bool,
        exit_ts: float | None = None,
    ) -> PaperPosition | None:
        """Settle a binary outcome: win → $1/share, lose → $0 (no exit fee)."""
        for pos in self.positions:
            if pos.position_id != position_id or pos.status != "open":
                continue
            exit_price = 1.0 if token_won else 0.0
            proceeds = exit_price * pos.shares
            pnl = proceeds - pos.notional - pos.fees
            self.cash += proceeds
            self.realized_pnl += pnl
            pos.status = "resolved"
            pos.exit_reason = "resolution"
            pos.exit_ts = exit_ts if exit_ts is not None else time.time()
            pos.exit_price = exit_price
            pos.exit_fees = 0.0
            pos.realized_pnl = pnl
            pos.win = token_won
            pos.mark_price = exit_price
            self.resolutions += 1
            self.persist()
            return pos
        return None

    def on_market_resolved(
        self,
        *,
        condition_id: str,
        winning_asset_id: str | None,
        exit_ts: float | None = None,
    ) -> list[PaperPosition]:
        """Settle all open paper positions for a resolved condition."""
        if not winning_asset_id:
            return []
        closed: list[PaperPosition] = []
        for pos in list(self.open_positions):
            if condition_id and pos.condition_id and pos.condition_id != condition_id:
                continue
            # If condition_id empty, only settle exact winning token (and skip others)
            if not condition_id and pos.token_id != winning_asset_id:
                continue
            won = pos.token_id == winning_asset_id
            # Same-condition losers settle too when condition_id is known
            if condition_id and pos.condition_id == condition_id:
                result = self.resolve_position(
                    pos.position_id, token_won=won, exit_ts=exit_ts
                )
                if result is not None:
                    closed.append(result)
            elif pos.token_id == winning_asset_id:
                result = self.resolve_position(
                    pos.position_id, token_won=True, exit_ts=exit_ts
                )
                if result is not None:
                    closed.append(result)
        return closed

    def maybe_poll_resolutions(self, *, now: float | None = None) -> list[PaperPosition]:
        """Periodically query Gamma for settled markets covering open positions."""
        now = time.time() if now is None else now
        if now - self._last_resolve_poll < self.config.resolve_poll_sec:
            return []
        self._last_resolve_poll = now
        closed: list[PaperPosition] = []
        seen_conditions: set[str] = set()
        for pos in list(self.open_positions):
            cid = pos.condition_id
            if not cid or cid in seen_conditions:
                continue
            seen_conditions.add(cid)
            info = fetch_market_resolution(cid)
            if info is None or not info.get("resolved"):
                continue
            winner = info.get("winning_token_id")
            closed.extend(
                self.on_market_resolved(
                    condition_id=cid,
                    winning_asset_id=str(winner) if winner else None,
                    exit_ts=now,
                )
            )
        return closed

    def snapshot(self) -> dict[str, Any]:
        return {
            "updated_at": utc_now_iso(),
            "bankroll_initial": self.config.bankroll,
            "cash": self.cash,
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "ticks_seen": self.ticks_seen,
            "signals_fired": self.signals_fired,
            "fills": self.fills,
            "resolutions": self.resolutions,
            "fee_category": self.config.fee_category,
            "fee_rate": self.config.effective_fee_rate(),
            "use_dynamic_fees": self.config.use_dynamic_fees,
            "n_strategies": len(self.strategies),
            "strategies": [
                {
                    "label": p.label(),
                    "oos_ev_pct": ev,
                    "oos_win_rate": wr,
                    "params": asdict(p),
                }
                for p, ev, wr in self.strategies
            ],
            "open_positions": [p.to_dict() for p in self.open_positions],
            "closed_positions": [
                p.to_dict() for p in self.positions if p.status != "open"
            ],
            "recent_signals": [asdict(s) for s in self.signals[-20:]],
        }

    def persist(self) -> None:
        if self.journal_path is None:
            return
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.journal_path.write_text(
            json.dumps(self.snapshot(), indent=2, default=str),
            encoding="utf-8",
        )


def fetch_market_resolution(
    condition_id: str,
    *,
    timeout_sec: float = 10.0,
) -> dict[str, Any] | None:
    """
    Query Gamma for market settlement state.

    Returns ``{resolved, winning_token_id, closed, raw}`` or None on failure.
    """
    if not condition_id:
        return None
    qs = urllib.parse.urlencode({"condition_ids": condition_id})
    url = f"{GAMMA_MARKETS_URL}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "polymarket-analytics/0.5"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None

    markets = raw if isinstance(raw, list) else [raw] if isinstance(raw, Mapping) else []
    if not markets:
        return None
    m = markets[0]
    if not isinstance(m, Mapping):
        return None

    closed = bool(m.get("closed") or m.get("resolved"))
    uma = str(m.get("umaResolutionStatus") or "").lower()
    resolved_flag = closed or uma in {"resolved", "settled"}

    winning_token: str | None = None
    tokens = m.get("tokens") or []
    if isinstance(tokens, list):
        for t in tokens:
            if isinstance(t, Mapping) and t.get("winner") in (True, "true", 1, "1"):
                winning_token = str(t.get("token_id") or t.get("tokenId") or "").strip()
                if winning_token:
                    break

    # outcomePrices like ["1","0"] paired with clobTokenIds
    if winning_token is None:
        prices = m.get("outcomePrices")
        clob = m.get("clobTokenIds") or m.get("clob_token_ids")
        price_list: list[str] = []
        token_list: list[str] = []
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                prices = []
        if isinstance(clob, str):
            try:
                clob = json.loads(clob)
            except json.JSONDecodeError:
                clob = []
        if isinstance(prices, list):
            price_list = [str(x) for x in prices]
        if isinstance(clob, list):
            token_list = [str(x) for x in clob]
        for px, tid in zip(price_list, token_list):
            try:
                if float(px) >= 0.99:
                    winning_token = tid
                    resolved_flag = True
                    break
            except ValueError:
                continue

    return {
        "resolved": bool(resolved_flag and winning_token),
        "winning_token_id": winning_token,
        "closed": closed,
        "raw": dict(m),
    }


def load_oos_strategies(
    report_path: Path | str | None,
    *,
    min_oos_ev_pct: float = 10.0,
    require_persists: bool = True,
) -> list[tuple[StrategyParams, float, float]]:
    """Load high-EV OOS setups from ``oos_edge_report.json``."""
    path = Path(report_path) if report_path else None
    loaded: list[tuple[StrategyParams, float, float]] = []
    if path is not None and path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            raw = {}
        if not isinstance(raw, Mapping):
            raw = {}
        for row in raw.get("comparisons") or []:
            if not isinstance(row, Mapping):
                continue
            oos_ev = float(row.get("oos_ev_pct") or 0.0)
            if oos_ev < min_oos_ev_pct:
                continue
            if require_persists and not row.get("persists", False):
                continue
            params_dict = row.get("params") or {}
            base = params_from_edge_row({**params_dict, **row})
            live_params = StrategyParams(
                price_bucket=base.price_bucket,
                min_volume_spike=None,
                min_whale_ratio=base.min_whale_ratio or 3.0,
                require_price_volume_divergence=base.require_price_volume_divergence,
                momentum_1h=base.momentum_1h,
                momentum_6h="any",
                max_time_to_resolution_hours=None,
                side=base.side or "BUY",
            )
            wr = float(row.get("oos_win_rate") or 0.5)
            loaded.append((live_params, oos_ev, wr))

    if loaded:
        seen: set[str] = set()
        uniq: list[tuple[StrategyParams, float, float]] = []
        for item in loaded:
            lab = item[0].label()
            if lab in seen:
                continue
            seen.add(lab)
            uniq.append(item)
        return uniq

    return [
        (p, max(min_oos_ev_pct, 15.0), 0.55) for p in DEFAULT_PAPER_STRATEGIES
    ]


def format_dashboard(
    trader: PaperTrader,
    *,
    feed_status: str = "idle",
    last_feat: LiveFeatures | None = None,
    extra_lines: Sequence[str] | None = None,
) -> str:
    """ANSI-friendly single-screen status block for the CLI."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(" POLYMARKET PAPER TRADER  (simulated fills only — no live orders)")
    lines.append("=" * 72)
    lines.append(
        f" feed={feed_status}  ticks={trader.ticks_seen}  "
        f"signals={trader.signals_fired}  fills={trader.fills}  "
        f"resolved={trader.resolutions}"
    )
    lines.append(
        f" cash=${trader.cash:,.2f}  equity=${trader.equity:,.2f}  "
        f"realized=${trader.realized_pnl:,.2f}  unrealized=${trader.unrealized_pnl:,.2f}"
    )
    lines.append(
        f" fees: category={trader.config.fee_category}  "
        f"rate={trader.config.effective_fee_rate():g}  "
        f"dynamic={trader.config.use_dynamic_fees}"
    )
    lines.append(f" strategies loaded: {len(trader.strategies)}")
    for p, ev, wr in trader.strategies[:5]:
        lines.append(f"   • {p.label()[:56]}  OOS EV={ev:.1f}pp  WR={100*wr:.0f}%")

    if last_feat is not None:
        wr = last_feat.whale_ratio
        mom = last_feat.momentum_1h
        whale_s = f"{wr:.2f}" if wr is not None else "n/a"
        mom_s = f"  mom1h={mom:+.4f}" if mom is not None else ""
        lines.append("-" * 72)
        lines.append(
            f" last tick  token={last_feat.token_id[:12]}…  "
            f"px={last_feat.price:.3f}  bucket={price_bucket_label(last_feat.price)}  "
            f"whale={whale_s}{mom_s}"
        )

    opens = trader.open_positions
    lines.append("-" * 72)
    lines.append(f" open positions ({len(opens)}):")
    if not opens:
        lines.append("   (none)")
    else:
        for pos in opens[-8:]:
            lines.append(
                f"   [{pos.position_id}] {pos.token_id[:10]}…  "
                f"fill={pos.entry_price_fill:.3f}  mark={pos.mark_price:.3f}  "
                f"fee=${pos.fees:.4f}  uPnL=${pos.unrealized_pnl:+.2f}  "
                f"{pos.strategy_label[:28]}"
            )

    if trader.signals:
        s = trader.signals[-1]
        lines.append(
            f" last signal: {s.strategy_label[:40]} @ {s.price:.3f} "
            f"(EV={s.oos_ev_pct:.1f}pp net={s.net_ev_pct:.1f}pp)"
        )

    if extra_lines:
        lines.extend(extra_lines)
    lines.append("=" * 72)
    return "\n".join(lines)
