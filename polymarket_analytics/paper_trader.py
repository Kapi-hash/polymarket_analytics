"""Phase 4: paper trading engine over live feature ticks (no real orders)."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from polymarket_analytics.backtest import StrategyParams, params_from_edge_row
from polymarket_analytics.live_feed import LiveFeatures
from polymarket_analytics.schema import PRICE_BUCKET_BREAKS, PRICE_BUCKET_LABELS

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
    # left-closed cuts matching Polars cut(left_closed=True)
    breaks = PRICE_BUCKET_BREAKS
    labels = PRICE_BUCKET_LABELS
    for i, b in enumerate(breaks):
        if price < b:
            return labels[i]
    return labels[-1]


def kelly_fraction(
    win_rate: float,
    *,
    payout_odds: float = 1.0,
    fraction: float = 0.25,
    max_fraction: float = 0.05,
) -> float:
    """
    Fractional Kelly for binary contracts.

    For a contract bought at price ``p``, payout on win is ``(1-p)/p`` in return
    terms (odds ``b``). Here ``payout_odds`` is that ``b``; when unknown we
    default to 1.0 (even money) and still apply a small fraction cap.
    """
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
    fee_bps: float = 0.0  # taker fee in basis points of notional
    spread_slippage_bps: float = 50.0  # half-spread / adverse selection
    min_oos_ev_pct: float = 10.0  # --min-ev 0.10 → 10 percentage points
    require_persists: bool = True
    max_open_positions: int = 25
    cooldown_sec: float = 60.0  # per token re-entry cooldown


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
    status: str = "open"  # open | resolved
    exit_ts: float | None = None
    exit_price: float | None = None
    realized_pnl: float | None = None
    win: bool | None = None

    @property
    def unrealized_pnl(self) -> float:
        if self.status != "open":
            return 0.0
        return (self.mark_price - self.entry_price_fill) * self.shares

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


@dataclass
class PaperTrader:
    """Evaluate live features vs OOS strategies; simulate fills only."""

    config: PaperConfig = field(default_factory=PaperConfig)
    strategies: list[tuple[StrategyParams, float, float]] = field(default_factory=list)
    # (params, oos_ev_pct, oos_win_rate)
    positions: list[PaperPosition] = field(default_factory=list)
    signals: list[SignalEvent] = field(default_factory=list)
    journal_path: Path | None = None
    cash: float = 0.0
    realized_pnl: float = 0.0
    ticks_seen: int = 0
    signals_fired: int = 0
    fills: int = 0
    _last_entry_ts: dict[str, float] = field(default_factory=dict)

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
        # Live feed has no volume_spike / TTR reliably — ignore those axes when unset data
        if params.min_volume_spike is not None:
            # Without live spike, skip strategies that hard-require it unless whale substitutes
            # Treat missing spike as non-match for strict OOS strategies that need it.
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
        """Model buy slippage: pay ask or raw + spread bps."""
        slip = self.config.spread_slippage_bps / 10_000.0
        fee = self.config.fee_bps / 10_000.0
        base = best_ask if best_ask is not None else raw_price
        fill = min(max(base * (1.0 + slip), 0.01), 0.99)
        # Fee modeled as worsening entry (effective cost)
        fill = min(fill * (1.0 + fee), 0.99)
        return fill

    def on_features(self, feat: LiveFeatures) -> PaperPosition | None:
        """Mark open positions and maybe open a new paper fill."""
        self.ticks_seen += 1
        for pos in self.open_positions:
            if pos.token_id == feat.token_id:
                pos.mark_price = feat.price

        if len(self.open_positions) >= self.config.max_open_positions:
            return None
        last = self._last_entry_ts.get(feat.token_id, 0.0)
        if feat.ts - last < self.config.cooldown_sec:
            return None
        # One open position per token
        if any(p.token_id == feat.token_id for p in self.open_positions):
            return None

        for params, oos_ev, oos_wr in self.strategies:
            if not self.matches(feat, params):
                continue
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
            )
            self.signals.append(sig)

            b = (1.0 - feat.price) / max(feat.price, 1e-9)
            f = kelly_fraction(
                oos_wr,
                payout_odds=b,
                fraction=self.config.kelly_fraction,
                max_fraction=self.config.max_position_pct,
            )
            if f <= 0:
                continue
            notional = self.equity * f
            if notional < 1.0 or notional > self.cash:
                continue

            fill_px = self.apply_fill_price(feat.price, best_ask=feat.best_ask)
            shares = notional / fill_px
            fees = notional * (self.config.fee_bps / 10_000.0)
            cost = notional + fees
            if cost > self.cash:
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
            )
            self.positions.append(pos)
            self._last_entry_ts[feat.token_id] = feat.ts
            self.fills += 1
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
        """Settle a binary outcome: win → $1/share, lose → $0."""
        for pos in self.positions:
            if pos.position_id != position_id or pos.status != "open":
                continue
            exit_price = 1.0 if token_won else 0.0
            proceeds = exit_price * pos.shares
            pnl = proceeds - pos.notional - pos.fees
            self.cash += proceeds
            self.realized_pnl += pnl
            pos.status = "resolved"
            pos.exit_ts = exit_ts if exit_ts is not None else time.time()
            pos.exit_price = exit_price
            pos.realized_pnl = pnl
            pos.win = token_won
            pos.mark_price = exit_price
            self.persist()
            return pos
        return None

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
                p.to_dict() for p in self.positions if p.status == "resolved"
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


def load_oos_strategies(
    report_path: Path | str | None,
    *,
    min_oos_ev_pct: float = 10.0,
    require_persists: bool = True,
) -> list[tuple[StrategyParams, float, float]]:
    """
    Load high-EV OOS setups from ``oos_edge_report.json``.

    Returns list of (params, oos_ev_pct, oos_win_rate). Falls back to defaults
    that only need live whale/price/momentum (no spike/TTR).
    """
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
            # Prefer strategies evaluable live (no hard spike/TTR requirement)
            # Soften: drop spike/TTR constraints so live engine can fire on whale+bucket
            base = params_from_edge_row({**params_dict, **row})
            live_params = StrategyParams(
                price_bucket=base.price_bucket,
                min_volume_spike=None,  # not available live yet
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
        # Deduplicate by label
        seen: set[str] = set()
        uniq: list[tuple[StrategyParams, float, float]] = []
        for item in loaded:
            lab = item[0].label()
            if lab in seen:
                continue
            seen.add(lab)
            uniq.append(item)
        return uniq

    # Fallback defaults (user-requested whale + mid-bucket)
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
        f"signals={trader.signals_fired}  fills={trader.fills}"
    )
    lines.append(
        f" cash=${trader.cash:,.2f}  equity=${trader.equity:,.2f}  "
        f"realized=${trader.realized_pnl:,.2f}  unrealized=${trader.unrealized_pnl:,.2f}"
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
                f"uPnL=${pos.unrealized_pnl:+.2f}  {pos.strategy_label[:32]}"
            )

    if trader.signals:
        s = trader.signals[-1]
        lines.append(
            f" last signal: {s.strategy_label[:40]} @ {s.price:.3f} "
            f"(EV={s.oos_ev_pct:.1f}pp)"
        )

    if extra_lines:
        lines.extend(extra_lines)
    lines.append("=" * 72)
    return "\n".join(lines)
